import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from application.config import RuntimeConfig
from application.recognition.landmarks import LandmarkDetector, LandmarkObservation
from application.recognition.classifier import CheckpointClassifier
from application.recognition.pipeline import RecognitionPipeline
from application.types import Candidate, FramePacket, PersonBox
from models.model_registry import ModelRegistry


class TinyImageModel(nn.Module):
    INPUT_FORMAT = "image"

    def __init__(self, class_count, pretrained=True):
        super().__init__()
        self.logits = nn.Parameter(torch.arange(class_count, dtype=torch.float32))
        self.pretrained = pretrained
        self.inference_mode = False

    def forward(self, inputs):
        self.inference_mode = torch.is_inference_mode_enabled()
        return self.logits.unsqueeze(0).expand(len(inputs), -1)


class TinyProbabilitiesModel(TinyImageModel):
    OUTPUT_FORMAT = "probabilities"

    def forward_logits(self, inputs):
        return super().forward(inputs)

    def forward(self, inputs):
        return self.forward_logits(inputs).softmax(dim=1)


class TinySequenceModel(TinyImageModel):
    INPUT_FORMAT = "sequence"


class CheckpointClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "final.pt"
        self.mapping = {"03": 2, "01": 0, "02": 1}
        self.checkpoint = {
            "config": {
                "model": "tiny_live", "processor": "landmarks",
                "model_options": {"pretrained": True},
                "processor_options": {
                    "subset": "asl_2nd", "imputation": True,
                    "anchor": "shoulders", "image_size": 16,
                },
            },
            "class_to_index": self.mapping,
            "model_state_dict": TinyImageModel(3).state_dict(),
        }
        torch.save(self.checkpoint, self.path)
        registry = patch.dict(ModelRegistry.MODELS, {"tiny_live": TinyImageModel})
        registry.start()
        self.addCleanup(registry.stop)

    def _write_labels(self, rows=None):
        with (self.directory / "per_class.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["class_index", "class_label"])
            writer.writerows(rows or [(2, "03 · Amarelo"), (0, "01 · Acontecer"), (1, "02 · Aluno")])

    def test_loads_only_checkpoint_weights_and_reads_labels_in_checkpoint_order(self):
        self._write_labels()
        classifier = CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))
        self.assertFalse(classifier._model.pretrained)
        self.assertFalse(classifier._model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in classifier._model.parameters()))
        points = np.full((7, classifier.processor.landmark_count, 2), 0.5, dtype=np.float32)
        candidates = classifier.predict(points)
        self.assertEqual([candidate.class_id for candidate in candidates], ["03", "02", "01"])
        self.assertEqual([candidate.label for candidate in candidates], ["Amarelo", "Aluno", "Acontecer"])
        expected = torch.arange(3, dtype=torch.float32).softmax(dim=0).numpy()[::-1]
        np.testing.assert_allclose([candidate.confidence for candidate in candidates], expected)
        self.assertTrue(classifier._model.inference_mode)
        self.assertFalse((self.directory / "processed").exists())

    def test_missing_labels_falls_back_to_class_ids_without_guessing_names(self):
        classifier = CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))
        self.assertEqual(classifier._labels, ("01", "02", "03"))

    def test_probability_models_do_not_receive_a_second_softmax(self):
        with patch.dict(ModelRegistry.MODELS, {"tiny_live": TinyProbabilitiesModel}):
            classifier = CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))
            points = np.full((7, classifier.processor.landmark_count, 2), 0.5, dtype=np.float32)
            candidates = classifier.predict(points)
            expected = torch.arange(3, dtype=torch.float32).softmax(dim=0).numpy()[::-1]
            np.testing.assert_allclose([candidate.confidence for candidate in candidates], expected)

    def test_sequence_models_reuse_the_same_landmark_preparation(self):
        self.checkpoint["config"]["processor_options"].update(representation="sequence", frame_count=8)
        torch.save(self.checkpoint, self.path)
        with patch.dict(ModelRegistry.MODELS, {"tiny_live": TinySequenceModel}):
            classifier = CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))
            points = np.full((7, classifier.processor.landmark_count, 2), 0.5, dtype=np.float32)
            self.assertEqual(classifier.predict(points)[0].class_id, "03")

    def test_incompatible_labels_fail_instead_of_misidentifying_the_sign(self):
        self._write_labels([(0, "02 · Aluno"), (1, "01 · Acontecer"), (2, "03 · Amarelo")])
        with self.assertRaisesRegex(ValueError, "ordem do checkpoint"):
            CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))

    def test_missing_class_or_duplicate_index_in_labels_is_rejected(self):
        self._write_labels([(0, "01 · Acontecer"), (0, "01 · Acontecer"), (2, "03 · Amarelo")])
        with self.assertRaisesRegex(ValueError, "ordem do checkpoint"):
            CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))
        self._write_labels([(0, "01 · Acontecer")])
        with self.assertRaisesRegex(ValueError, "mesmas classes"):
            CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))

    def test_unknown_processor_and_incompatible_representation_have_clear_errors(self):
        self.checkpoint["config"]["processor"] = "other"
        torch.save(self.checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "processor='landmarks'"):
            CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))
        self.checkpoint["config"]["processor"] = "landmarks"
        self.checkpoint["config"]["processor_options"]["representation"] = "sequence"
        torch.save(self.checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "recebe 'image'"):
            CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))

    def test_class_mapping_must_be_contiguous(self):
        self.checkpoint["class_to_index"] = {"01": 0, "02": 2, "03": 3}
        torch.save(self.checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "class_to_index"):
            CheckpointClassifier(RuntimeConfig(checkpoint=self.path, device="cpu"))


class FakeClassifier:
    def __init__(self):
        self.device = "cpu"
        self.processor = SimpleNamespace(
            landmark_groups=(("pose", (11, 12)), ("left_hand", (0, 1)), ("right_hand", (0, 1))),
            anchor_indices=[0, 1],
        )
        self.windows = []
        self.candidates = (
            Candidate("01", "Acontecer", 0.9),
            Candidate("02", "Aluno", 0.07),
            Candidate("03", "Amarelo", 0.03),
        )

    def predict(self, window):
        self.windows.append(window.copy())
        return self.candidates


class FakeDetector:
    def __init__(self):
        points = np.array([[0.4, 0.4], [0.6, 0.4], [0.4, 0.7], [0.45, 0.7], [0.6, 0.7], [0.65, 0.7]])
        self.observation = LandmarkObservation(points, PersonBox(0.2, 0.1, 0.8, 0.9), True)
        self.closed = False
        self.resets = 0
        self.images = []

    def process(self, image):
        self.images.append(image)
        return self.observation

    def close(self):
        self.closed = True

    def reset(self):
        self.resets += 1


class RecognitionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = RuntimeConfig(
            window_seconds=0.3, sample_fps=10, prediction_interval=0.1,
            max_frame_gap=0.2, person_timeout=0.2, stability_count=2,
        )
        self.classifier = FakeClassifier()
        self.detector = FakeDetector()
        self.created = []

        def create_classifier(config):
            self.created.append("classifier")
            return self.classifier

        def create_detector(groups):
            self.created.append("detector")
            self.assertEqual(groups, self.classifier.processor.landmark_groups)
            return self.detector

        self.pipeline = RecognitionPipeline(
            self.config, classifier_factory=create_classifier, detector_factory=create_detector,
        )
        self.image = np.zeros((8, 12, 3), dtype=np.uint8)

    def frame(self, timestamp):
        return FramePacket(round(timestamp * 100), timestamp, self.image)

    def warm_up(self):
        for index in range(5):
            result = self.pipeline.process(self.frame(index / 10))
        return result

    def test_heavy_resources_are_created_lazily_and_always_closed(self):
        self.assertEqual(self.created, [])
        with self.assertRaisesRegex(RuntimeError, "with"):
            self.pipeline.process(self.frame(0))
        with self.pipeline:
            self.assertEqual(self.created, ["classifier", "detector"])
            self.assertEqual(self.pipeline.device, "cpu")
            self.warm_up()
        self.assertTrue(self.detector.closed)

    def test_waits_for_complete_window_then_repeated_predictions_before_caption(self):
        with self.pipeline:
            for timestamp in (0, 0.1, 0.2):
                result = self.pipeline.process(self.frame(timestamp))
                self.assertEqual(result.candidates, ())
            result = self.pipeline.process(self.frame(0.3))
            self.assertEqual(len(result.candidates), 3)
            self.assertEqual(result.caption, "")
            self.assertEqual(result.progress, 1)
            result = self.pipeline.process(self.frame(0.35))
            self.assertEqual(len(self.classifier.windows), 1)
            self.assertEqual(result.caption, "")
            result = self.pipeline.process(self.frame(0.4))
            self.assertEqual(len(self.classifier.windows), 2)
            self.assertEqual(result.caption, "Acontecer")
            self.assertEqual(result.prediction_timestamp, 0.4)
            self.assertIs(self.detector.images[0], self.image)

    def test_losing_hands_removes_caption_without_erasing_the_person_context(self):
        with self.pipeline:
            self.assertEqual(self.warm_up().caption, "Acontecer")
            previous = self.detector.observation
            points = previous.points.copy()
            points[2:] = np.nan
            self.detector.observation = LandmarkObservation(points, previous.box, False)
            missing = self.pipeline.process(self.frame(0.5))
            self.assertEqual(missing.caption, "")
            self.assertEqual(missing.candidates, ())
            self.assertEqual(missing.progress, 1)
            self.assertEqual(missing.box, previous.box)
            self.assertEqual(len(self.classifier.windows), 2)
            self.detector.observation = previous
            result = self.pipeline.process(self.frame(0.6))
            self.assertEqual(len(result.candidates), 3)
            self.assertEqual(result.caption, "")
            self.assertEqual(result.progress, 1)
            self.assertEqual(len(self.classifier.windows), 3)
            self.assertTrue(np.isnan(self.classifier.windows[-1][-2, 2:]).all())

    def test_prefix_without_hands_fills_window_but_never_runs_inference(self):
        with self.pipeline:
            visible = self.detector.observation
            points = visible.points.copy()
            points[2:] = np.nan
            self.detector.observation = LandmarkObservation(points, visible.box, False)
            for timestamp in (0, 0.1, 0.2, 0.3, 0.4):
                result = self.pipeline.process(self.frame(timestamp))
                self.assertEqual(result.candidates, ())
                self.assertEqual(result.caption, "")
            self.assertEqual(result.progress, 1)
            self.assertEqual(len(self.classifier.windows), 0)
            self.detector.observation = visible
            result = self.pipeline.process(self.frame(0.5))
            self.assertEqual(len(result.candidates), 3)
            self.assertEqual(len(self.classifier.windows), 1)
            self.assertTrue(np.isnan(self.classifier.windows[0][:-1, 2:]).all())
            self.assertTrue(np.isfinite(self.classifier.windows[0][-1]).all())

    def test_person_timeout_clears_tracking_once_and_does_not_keep_stale_predictions(self):
        with self.pipeline:
            self.warm_up()
            self.detector.observation = LandmarkObservation(np.zeros((6, 2)), None, False)
            for timestamp in (0.5, 0.7, 0.8):
                result = self.pipeline.process(self.frame(timestamp))
                self.assertEqual(result.caption, "")
                self.assertEqual(result.candidates, ())
                self.assertIsNone(result.box)
            self.assertEqual(self.detector.resets, 1)

    def test_large_timestamp_gap_restarts_even_if_person_remains_visible(self):
        with self.pipeline:
            self.warm_up()
            result = self.pipeline.process(self.frame(1))
            self.assertEqual(result.candidates, ())
            self.assertEqual(result.caption, "")
            self.assertEqual(result.progress, 0)

    def test_missing_shoulders_prevents_unanchored_predictions(self):
        with self.pipeline:
            self.warm_up()
            previous = self.detector.observation
            points = previous.points.copy()
            points[0] = np.nan
            self.detector.observation = LandmarkObservation(points, previous.box, True)
            result = self.pipeline.process(self.frame(0.5))
            self.assertEqual(result.candidates, ())
            self.assertEqual(result.caption, "")
            self.assertIn("Enquadre", result.status)

    def test_low_confidence_or_different_word_does_not_keep_previous_caption(self):
        with self.pipeline:
            self.warm_up()
            self.classifier.candidates = (Candidate("02", "Aluno", 0.6),)
            result = self.pipeline.process(self.frame(0.5))
            self.assertEqual(result.caption, "")
            self.classifier.candidates = (Candidate("02", "Aluno", 0.95),)
            self.assertEqual(self.pipeline.process(self.frame(0.6)).caption, "")
            self.assertEqual(self.pipeline.process(self.frame(0.7)).caption, "Aluno")

    def test_reset_and_body_jump_do_not_mix_windows_from_different_people(self):
        with self.pipeline:
            self.detector.observation = LandmarkObservation(
                self.detector.observation.points, PersonBox(0, 0.1, 0.3, 0.9), True,
            )
            self.warm_up()
            self.detector.observation = LandmarkObservation(
                self.detector.observation.points + 0.4, PersonBox(0.7, 0.1, 1, 0.9), True,
            )
            result = self.pipeline.process(self.frame(0.5))
            self.assertEqual(result.caption, "")
            self.assertEqual(result.candidates, ())
            self.assertEqual(result.progress, 0)
            self.pipeline.reset()
            self.assertEqual(self.detector.resets, 1)
            result = self.pipeline.process(self.frame(0))
            self.assertEqual(result.progress, 0)


class LandmarkDetectorTests(unittest.TestCase):
    def test_selection_matches_training_order_and_original_bgr_is_never_mirrored(self):
        pose = [SimpleNamespace(x=index / 100, y=index / 100, visibility=1) for index in range(33)]
        detected = SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=pose), face_landmarks=None,
            left_hand_landmarks=None, right_hand_landmarks=SimpleNamespace(landmark=pose[:21]),
        )
        holistic = SimpleNamespace(
            process=lambda image: (setattr(holistic, "image", image), detected)[1],
            close=lambda: None, reset=lambda: None,
        )
        constructor = unittest.mock.Mock(return_value=holistic)
        module = SimpleNamespace(solutions=SimpleNamespace(holistic=SimpleNamespace(Holistic=constructor)))
        with patch.dict(sys.modules, {"mediapipe": module}):
            detector = LandmarkDetector((("pose", (12, 11)), ("left_hand", (1, 0))))
            image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
            observation = detector.process(image)
            np.testing.assert_allclose(observation.points[:2], [[0.12, 0.12], [0.11, 0.11]])
            self.assertTrue(np.isnan(observation.points[2:]).all())
            np.testing.assert_array_equal(holistic.image, image[:, :, ::-1])
            self.assertTrue(observation.hands_present)
            constructor.assert_called_once_with(
                min_detection_confidence=0.4, min_tracking_confidence=0.4,
                refine_face_landmarks=False, model_complexity=1,
            )
            detector.reset()
            detector.close()

    def test_person_box_ignores_low_visibility_pose_points_and_clips_to_screen(self):
        points = [SimpleNamespace(x=0.2, y=0.1, visibility=1), SimpleNamespace(x=0.8, y=1.1, visibility=1)]
        points.append(SimpleNamespace(x=-100, y=-100, visibility=0))
        detected = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points), face_landmarks=None)
        box = LandmarkDetector._person_box(detected)
        self.assertAlmostEqual(box.left, 0.175)
        self.assertAlmostEqual(box.top, 0.075)
        self.assertAlmostEqual(box.right, 0.825)
        self.assertEqual(box.bottom, 1)


if __name__ == "__main__":
    unittest.main()
