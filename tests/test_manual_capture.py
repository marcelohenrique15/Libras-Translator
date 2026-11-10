from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from application.capture.session import CaptureSession
from application.config import RuntimeConfig
from application.recognition.clip_pipeline import ClipRecognitionPipeline
from application.types import AnalysisResult, Candidate, FramePacket, PersonBox
from application.ui.capture_overlay import CaptureRenderer


class ClipClassifier:
    device = "cpu"
    processor = SimpleNamespace(landmark_groups=(("pose", (11, 12)), ("left_hand", (0, 1))),
                                anchor_indices=[0, 1])

    def __init__(self):
        self.windows = []
        self.confidence = .95

    def predict(self, window):
        self.windows.append(window.copy())
        return (Candidate("01", "Acontecer", self.confidence), Candidate("02", "Aluno", 1 - self.confidence))


class ClipDetector:
    def __init__(self):
        self.points = np.ones((4, 2), dtype=np.float32)
        self.hands = True
        self.closed = False
        self.calls = 0
        self.resets = 0

    def process(self, image):
        self.calls += 1
        return SimpleNamespace(points=self.points.copy(), box=PersonBox(.1, .1, .9, .9),
                               hands_present=self.hands)

    def close(self):
        self.closed = True

    def reset(self):
        self.resets += 1


class ManualCaptureTests(unittest.TestCase):
    def setUp(self):
        self.config = RuntimeConfig(mode="manual", capture_seconds=1, max_frame_gap=.25)
        self.classifier = ClipClassifier()
        self.detector = ClipDetector()
        self.pipeline = ClipRecognitionPipeline(self.config, classifier_factory=lambda _: self.classifier,
                                                detector_factory=lambda _: self.detector)
        self.frame = FramePacket(0, 10, np.zeros((240, 320, 3), dtype=np.uint8))

    def record(self):
        self.pipeline.start_capture(10)
        for index in range(10):
            result = self.pipeline.process(replace(self.frame, frame_id=index, timestamp=10 + index / 10))
            self.assertEqual(result.capture_state, "recording")
            self.assertFalse(result.candidates)
            self.assertEqual(len(self.classifier.windows), 0)

    def test_no_model_inference_before_trigger_or_before_end_and_only_one_after_end(self):
        with self.pipeline:
            self.assertEqual(self.pipeline.process(self.frame).capture_state, "idle")
            self.record()
            result = self.pipeline.process(replace(self.frame, timestamp=11))
            self.assertEqual(result.capture_state, "completed")
            self.assertEqual(result.caption, "Acontecer")
            self.assertEqual(len(self.classifier.windows), 1)
            self.assertEqual(self.detector.calls, 10)  # frame após o prazo não entra no vídeo
            self.assertIs(self.pipeline.process(replace(self.frame, timestamp=12)), result)
            self.assertEqual(len(self.classifier.windows), 1)
        self.assertTrue(self.detector.closed)

    def test_explicit_finish_and_new_capture_do_not_mix_previous_gesture(self):
        with self.pipeline:
            self.record()
            first = self.pipeline.finish_capture(replace(self.frame, timestamp=10.9))
            self.assertEqual(first.caption, "Acontecer")
            self.detector.points.fill(2)
            self.pipeline.start_capture(20)
            for index in range(10):
                self.pipeline.process(replace(self.frame, timestamp=20 + index / 10))
            self.pipeline.finish_capture(replace(self.frame, timestamp=21))
            np.testing.assert_allclose(self.classifier.windows[0], 1)
            np.testing.assert_allclose(self.classifier.windows[1], 2)
            self.assertEqual(self.detector.resets, 0)

    def test_missing_hands_short_recording_or_transport_gap_never_classifies(self):
        with self.pipeline:
            self.detector.hands = False
            self.record()
            self.assertEqual(self.pipeline.finish_capture(replace(self.frame, timestamp=11)).capture_state, "rejected")
            self.pipeline.start_capture(20)
            self.pipeline.process(replace(self.frame, timestamp=20))
            self.assertEqual(self.pipeline.finish_capture(replace(self.frame, timestamp=21)).capture_state, "rejected")
            self.detector.hands = True
            self.pipeline.start_capture(30)
            for offset in (0, .1, .2, .3, .4, .8, .85, .9, .95):
                self.pipeline.process(replace(self.frame, timestamp=30 + offset))
            self.assertEqual(self.pipeline.finish_capture(replace(self.frame, timestamp=31)).capture_state, "rejected")
            self.assertFalse(self.classifier.windows)

    def test_low_confidence_is_visible_as_uncertain_instead_of_success(self):
        self.classifier.confidence = .55
        with self.pipeline:
            self.record()
            result = self.pipeline.finish_capture(replace(self.frame, timestamp=11))
            self.assertEqual(result.caption, "Possível: Acontecer")
            self.assertTrue(result.caption_is_partial)
            self.assertIn("Baixa confiança", result.status)
            self.assertTrue(result.candidates)

    def test_early_stop_uses_actual_duration_without_padding_to_six_seconds(self):
        pipeline = ClipRecognitionPipeline(replace(self.config, capture_seconds=6),
                                           classifier_factory=lambda _: self.classifier,
                                           detector_factory=lambda _: self.detector)
        with pipeline:
            pipeline.start_capture(10)
            for index in range(12):
                pipeline.process(replace(self.frame, timestamp=10 + index / 10))
            result = pipeline.finish_capture(replace(self.frame, timestamp=11.15))
            self.assertEqual(result.capture_state, "completed")
            self.assertAlmostEqual(result.capture_metrics["duration_seconds"], 1.15)
            self.assertEqual(result.capture_metrics["landmark_frames"], 12)
            self.assertLess(len(self.classifier.windows[0]), 180)

    def test_session_ignores_repeated_trigger_and_retains_result_until_next_trigger(self):
        session = CaptureSession(6)
        self.assertFalse(session.busy)
        self.assertTrue(session.start(10))
        self.assertFalse(session.start(11))
        self.assertEqual(session.scene(15).capture_state, "recording")
        self.assertEqual(session.scene(16).capture_state, "processing")
        result = AnalysisResult(1, 16.1, None, caption="Aluno", capture_state="completed")
        session.accept(result)
        self.assertEqual(session.scene(40).caption, "Aluno")
        self.assertTrue(session.start(41))
        self.assertFalse(session.scene(41).caption)
        session.cancel()
        session.accept(result)
        self.assertEqual(session.scene(42).capture_state, "idle")

    def test_stop_freezes_end_time_and_can_only_be_requested_once(self):
        session = CaptureSession(6)
        self.assertFalse(session.stop(9))
        session.start(10)
        self.assertTrue(session.stop(12.5))
        self.assertEqual(session.ended_at, 12.5)
        self.assertFalse(session.stop(13))
        self.assertEqual(session.scene(14).capture_state, "processing")
        self.assertEqual(session.ended_at, 12.5)

    def test_caption_is_below_video_even_without_a_person_box_and_input_stays_unchanged(self):
        result = AnalysisResult(1, 10, None, caption="Aluno", capture_state="completed")
        renderer = CaptureRenderer(mirror=False)
        before = self.frame.image.copy()
        with patch.object(renderer, "_text", wraps=renderer._text) as text:
            image = renderer.render(self.frame, result)
        np.testing.assert_array_equal(self.frame.image, before)
        self.assertGreater(image.shape[0], before.shape[0])
        caption = next(call for call in text.call_args_list if call.args[1] == "Aluno")
        self.assertGreaterEqual(caption.args[3], before.shape[0])

    def test_defaults_enable_six_second_manual_capture(self):
        config = RuntimeConfig()
        self.assertEqual((config.mode, config.capture_seconds), ("manual", 6))
        for settings in ({"capture_seconds": 0}, {"capture_seconds": float("nan")}, {"mode": "other"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RuntimeConfig(**settings)
