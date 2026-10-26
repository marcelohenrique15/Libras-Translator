import unittest
from pathlib import Path

import numpy as np

from application.recognition.temporal import LandmarkWindow, PredictionStability
from application.types import Candidate


class LandmarkWindowTests(unittest.TestCase):
    def test_irregular_frames_are_resampled_using_the_actual_clock(self):
        window = LandmarkWindow(seconds=1, sample_fps=6, max_gap=0.6)
        for timestamp in (0, 0.1, 0.45, 0.8, 1):
            window.append(timestamp, np.array([[timestamp, timestamp * 2]]))
        self.assertTrue(window.ready)
        result = window.resample()
        np.testing.assert_allclose(result[:, 0, 0], np.linspace(0, 1, 6))
        np.testing.assert_allclose(result[:, 0, 1], np.linspace(0, 2, 6))

    def test_missing_detection_is_not_interpolated_into_a_phantom_hand(self):
        window = LandmarkWindow(seconds=1, sample_fps=6, max_gap=0.6)
        window.append(0, np.array([[0, 0], [1, 1]]))
        window.append(0.5, np.array([[np.nan, np.nan], [2, 2]]))
        window.append(1, np.array([[1, 1], [3, 3]]))
        result = window.resample()
        self.assertTrue(np.isnan(result[1:-1, 0]).all())
        np.testing.assert_array_equal(result[[0, -1], 0], [[0, 0], [1, 1]])
        np.testing.assert_allclose(result[:, 1, 0], np.linspace(1, 3, 6))

    def test_large_gap_starts_a_fresh_window_and_drops_previous_positions(self):
        window = LandmarkWindow(seconds=1, sample_fps=5, max_gap=0.3)
        for timestamp in (0, 0.2, 0.4):
            self.assertFalse(window.append(timestamp, np.array([[0, 0]])))
        self.assertTrue(window.append(1, np.array([[9, 9]])))
        self.assertFalse(window.ready)
        self.assertEqual(window.progress, 0)
        for timestamp in (1.2, 1.4, 1.6, 1.8, 2):
            window.append(timestamp, np.array([[9, 9]]))
        np.testing.assert_array_equal(window.resample(), np.full((6, 1, 2), 9))

    def test_window_has_bounded_memory_even_if_source_exceeds_expected_fps(self):
        window = LandmarkWindow(seconds=1, sample_fps=10, max_gap=0.5)
        for index in range(5000):
            window.append(index / 1000, np.array([[index, index]]))
        self.assertLessEqual(len(window._samples), 42)
        self.assertTrue(window.ready)
        np.testing.assert_allclose(window.resample()[:, 0, 0], np.linspace(3999, 4999, 9))

    def test_old_frames_are_removed_but_a_boundary_neighbor_is_kept(self):
        window = LandmarkWindow(seconds=1, sample_fps=4, max_gap=0.5)
        for timestamp in np.arange(0, 10.25, 0.25):
            window.append(float(timestamp), np.array([[timestamp, timestamp]]))
        np.testing.assert_allclose(window.resample()[:, 0, 0], [9, 9.5, 10])
        self.assertLessEqual(len(window._samples), 6)

    def test_nonmonotonic_frames_and_incomplete_windows_fail_clearly(self):
        window = LandmarkWindow(seconds=1, sample_fps=4, max_gap=0.5)
        window.append(1, np.array([[0, 0]]))
        for timestamp in (1, 0.5):
            with self.assertRaisesRegex(ValueError, "ordem temporal"):
                window.append(timestamp, np.array([[0, 0]]))
        with self.assertRaisesRegex(ValueError, "duração"):
            window.resample()

    def test_default_grid_keeps_the_newest_frame_after_the_actual_rgb_encoder(self):
        from preprocessor.streaming_processor import StreamingLandmarkProcessor

        processor = StreamingLandmarkProcessor(Path("."), imputation=False, anchor="none", image_size=60)
        window = LandmarkWindow(seconds=3, sample_fps=30, max_gap=0.35)
        for timestamp in np.linspace(0, 3, 91):
            points = np.full((processor.landmark_count, 2), timestamp / 3, dtype=np.float32)
            window.append(float(timestamp), points)
        resampled = window.resample()
        self.assertEqual(len(resampled), 90)
        np.testing.assert_array_equal(resampled[-1], np.ones((processor.landmark_count, 2)))
        encoded = processor.encode(processor.prepare_array(resampled))
        np.testing.assert_array_equal(encoded[2, :, -1], np.ones(60))


class PredictionStabilityTests(unittest.TestCase):
    def test_only_repeated_high_confidence_predictions_release_one_word(self):
        stability = PredictionStability(confidence=0.7, count=3)
        high = (Candidate("01", "Acontecer", 0.9),)
        low = (Candidate("01", "Acontecer", 0.6),)
        self.assertEqual(stability.update(high), "")
        self.assertEqual(stability.update(high), "")
        self.assertEqual(stability.update(high), "Acontecer")
        self.assertEqual(stability.update(low), "")
        self.assertEqual(stability.update(high), "")
        self.assertEqual(stability.update(()), "")

    def test_change_of_class_immediately_clears_the_previous_caption(self):
        stability = PredictionStability(confidence=0.7, count=2)
        first = (Candidate("01", "Acontecer", 0.9),)
        second = (Candidate("02", "Aluno", 0.95),)
        stability.update(first)
        self.assertEqual(stability.update(first), "Acontecer")
        self.assertEqual(stability.update(second), "")
        self.assertEqual(stability.update(second), "Aluno")
        stability.reset()
        self.assertEqual(stability.caption, "")


if __name__ == "__main__":
    unittest.main()
