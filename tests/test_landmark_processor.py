import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from preprocessor.landmark_processor import LandmarkProcessor


class LandmarkProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)
        self.processor = LandmarkProcessor(self.root, imputation=False)

    def _point_index(self, processor, group, point):
        names = [
            (group_name, point_number)
            for group_name, points in processor.landmark_groups
            for point_number in points
        ]
        return names.index((group, point))

    def _landmarks(self, frame_count=6):
        return np.full((frame_count, self.processor.landmark_count, 2), 0.5, dtype=np.float32)

    def _set_shoulders(self, landmarks, left, right):
        landmarks[:, self._point_index(self.processor, "pose", 11)] = left
        landmarks[:, self._point_index(self.processor, "pose", 12)] = right

    def _write_sample(self, sample_id, landmarks):
        columns = [
            f"{group}_{point}_{axis}"
            for group, points in self.processor.landmark_groups
            for point in points
            for axis in ("x", "y")
        ]
        table = pd.DataFrame(landmarks.reshape(len(landmarks), -1), columns=columns)
        table.insert(0, "frame", np.arange(len(landmarks)))
        path = self.root / "processed" / "landmarks" / f"{sample_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        return path

    def test_shoulder_anchor_removes_camera_translation_and_preserves_missing_values(self):
        landmarks = self._landmarks()
        self._set_shoulders(landmarks, [0.3, 0.4], [0.7, 0.4])
        hand = self._point_index(self.processor, "left_hand", 0)
        landmarks[:, hand] = [0.2, 0.3]
        landmarks[2, hand] = np.nan
        original = landmarks.copy()

        anchored = self.processor.anchor(landmarks)
        shifted = self.processor.anchor(landmarks + np.array([0.2, -0.1], dtype=np.float32))

        np.testing.assert_allclose(anchored, shifted, atol=2e-7, equal_nan=True)
        np.testing.assert_allclose(anchored[0, hand], [-0.3, -0.1], atol=2e-7)
        self.assertTrue(np.isnan(anchored[2, hand]).all())
        np.testing.assert_array_equal(landmarks, original)

    def test_fixed_shoulder_reference_preserves_head_and_hand_trajectories(self):
        landmarks = self._landmarks()
        motion = np.arange(6, dtype=np.float32) * 0.1
        self._set_shoulders(
            landmarks,
            np.column_stack((0.2 + motion, np.full(6, 0.4))),
            np.column_stack((0.4 + motion, np.full(6, 0.4))),
        )
        nose = self._point_index(self.processor, "pose", 0)
        hand = self._point_index(self.processor, "left_hand", 0)
        landmarks[:, nose, 0] = 0.3 + motion
        landmarks[:, hand] = [0.3, 0.2]

        anchored = self.processor.anchor(landmarks)

        # A origem fixa é a mediana dos centros: (0.55, 0.4).
        np.testing.assert_allclose(anchored[:, hand], np.tile([-0.25, -0.2], (6, 1)), atol=2e-7)
        np.testing.assert_allclose(
            np.diff(anchored[:, nose], axis=0), np.diff(landmarks[:, nose], axis=0), atol=2e-7,
        )
        self.assertGreater(float(np.ptp(anchored[:, nose, 0])), 0.4)

    def test_shoulder_reference_uses_only_frames_with_both_complete_shoulders(self):
        landmarks = self._landmarks(frame_count=3)
        self._set_shoulders(
            landmarks,
            [[0.2, 0.4], [0.8, 0.8], [0.4, 0.4]],
            [[0.4, 0.4], [np.nan, np.nan], [0.6, 0.4]],
        )
        hand = self._point_index(self.processor, "left_hand", 0)

        anchored = self.processor.anchor(landmarks)

        np.testing.assert_allclose(anchored[:, hand], np.full((3, 2), 0.1), atol=2e-7)

    def test_missing_anchor_has_an_explicit_error_instead_of_using_zero(self):
        landmarks = self._landmarks()
        self._set_shoulders(landmarks, [0.3, 0.4], [np.nan, np.nan])
        with self.assertRaisesRegex(ValueError, "(?i)shoulders|ombro"):
            self.processor.anchor(landmarks)

        nose_processor = LandmarkProcessor(self.root, anchor="nose")
        landmarks[:, self._point_index(nose_processor, "pose", 0)] = np.nan
        with self.assertRaisesRegex(ValueError, "(?i)nose|nariz"):
            nose_processor.anchor(landmarks)

    def test_nose_and_legacy_modes_are_explicit_options(self):
        landmarks = self._landmarks()
        nose_processor = LandmarkProcessor(self.root, anchor="nose")
        nose = self._point_index(nose_processor, "pose", 0)
        landmarks[:, nose] = [0.4, 0.3]
        np.testing.assert_allclose(nose_processor.anchor(landmarks)[:, nose], 0, atol=2e-7)
        np.testing.assert_allclose(
            nose_processor.anchor(landmarks)[:, 0], np.tile([0.1, 0.2], (6, 1)), atol=2e-7,
        )

        legacy = LandmarkProcessor(self.root, anchor="none")
        landmarks[0, 0] = np.nan
        np.testing.assert_array_equal(legacy.anchor(landmarks), landmarks)
        with self.assertRaisesRegex(ValueError, "(?i)anchor|ancor"):
            LandmarkProcessor(self.root, anchor="unknown")

    def test_rgb_encoding_reserves_black_for_absence_and_gray_for_the_origin(self):
        processor = LandmarkProcessor(self.root, image_size=2)
        # Dois pontos e três frames produzem uma imagem 2 x 2 sem redimensionamento.
        landmarks = np.full((3, 2, 2), np.nan, dtype=np.float32)
        landmarks[:, :, 0] = 0

        image = np.rint(processor.encode(landmarks) * 255).astype(np.uint8)

        np.testing.assert_array_equal(image[:, :, 0], np.full((3, 2), 128, dtype=np.uint8))
        np.testing.assert_array_equal(image[:, :, 1], np.zeros((3, 2), dtype=np.uint8))

    def test_rgb_encoding_clips_signed_coordinates_without_losing_missing_frames(self):
        processor = LandmarkProcessor(self.root, image_size=2)
        landmarks = np.full((3, 2, 2), np.nan, dtype=np.float32)
        landmarks[:, 0, 0] = [-2, 0, 2]
        landmarks[:, 1, 0] = [np.nan, 0, np.nan]

        image = np.rint(processor.encode(landmarks) * 255).astype(np.uint8)

        np.testing.assert_array_equal(image[:, 0, 0], [1, 128, 255])
        np.testing.assert_array_equal(image[:, 1, 0], [0, 128, 0])

    def test_prepare_and_image_cache_keep_absence_until_encoding(self):
        landmarks = self._landmarks()
        self._set_shoulders(landmarks, [0.3, 0.4], [0.7, 0.4])
        hand = self._point_index(self.processor, "left_hand", 0)
        landmarks[:, hand] = np.nan
        self._write_sample("sample", landmarks)

        prepared = self.processor.prepare("sample")
        self.assertTrue(np.isnan(prepared[:, hand]).all())
        np.testing.assert_allclose(prepared[:, 0], np.tile([0, 0.1], (6, 1)), atol=2e-7)
        image = self.processor.process("sample")
        self.assertTrue(np.isfinite(image).all())
        cache = np.load(self.processor.encoding_dir / "sample.npy", allow_pickle=False)
        self.assertEqual(cache.dtype, np.uint8)
        with patch.object(self.processor, "prepare", side_effect=AssertionError("cache não reutilizado")):
            np.testing.assert_array_equal(self.processor.process("sample"), image)

        imputed = LandmarkProcessor(self.root, imputation=True).prepare("sample")
        self.assertTrue(np.isnan(imputed[:, hand]).all())

    def test_sequence_keeps_signed_coordinates_and_handles_missing_values(self):
        processor = LandmarkProcessor(self.root, representation="sequence", frame_count=3)
        landmarks = np.array([[[-0.3, np.nan]], [[-0.1, np.nan]], [[0.2, np.nan]]], dtype=np.float32)

        sequence = processor.encode(landmarks)

        np.testing.assert_allclose(sequence[:, 0], [-0.3, -0.1, 0.2], atol=2e-7)
        np.testing.assert_array_equal(sequence[:, 1], np.zeros(3, dtype=np.float32))

    def test_encoded_cache_separates_anchor_options(self):
        processors = [LandmarkProcessor(self.root, anchor=anchor) for anchor in ("shoulders", "nose", "none")]
        self.assertEqual(len({processor.encoding_dir for processor in processors}), 3)
        # Seleção e interpolação são compartilháveis: ainda usam coordenadas da câmera.
        self.assertEqual(len({processor.selection_dir for processor in processors}), 1)
        self.assertEqual(len({processor.imputation_dir for processor in processors}), 1)


if __name__ == "__main__":
    unittest.main()
