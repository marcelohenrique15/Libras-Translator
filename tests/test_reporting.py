import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from training.reporting import ExperimentReporter


class ExperimentReporterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.reporter = ExperimentReporter(self.root, ["água", "dor", "médico"])
        self.history = [
            {"epoch": 1, "train_loss": 0.8, "train_objective": 0.85, "train_f1_macro": 0.5},
            {"epoch": 2, "train_loss": 0.6, "train_objective": 0.64, "train_f1_macro": 0.7},
        ]
        self.result = {
            "sample_ids": ["video_01", "video_02", "video_03", "video_04"],
            "metrics": {
                "loss": 0.5, "accuracy": 0.75, "f1_macro": 0.4888888889,
                "expected": [0, 0, 0, 1], "predicted": [0, 0, 1, 1],
                "probabilities": [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]],
                "confusion_matrix": [[2, 1, 0], [0, 1, 0], [0, 0, 0]],
            },
        }

    def _rows(self, filename):
        with (self.root / filename).open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def test_exports_predictions_rates_and_valid_png_without_figure_leaks(self):
        from matplotlib import pyplot

        initial_figures = pyplot.get_fignums()
        self.reporter.write(self.history, self.result)
        predictions = self._rows("predictions.csv")
        self.assertEqual([row["sample_id"] for row in predictions], self.result["sample_ids"])
        self.assertEqual(predictions[2]["expected_class"], "água")
        self.assertEqual(predictions[2]["predicted_class"], "dor")
        self.assertEqual(predictions[2]["correct"], "False")
        self.assertAlmostEqual(float(predictions[2]["confidence"]), 0.7)

        classes = self._rows("per_class.csv")
        self.assertEqual([int(row["support"]) for row in classes], [3, 1, 0])
        self.assertAlmostEqual(float(classes[0]["precision"]), 1.0)
        self.assertAlmostEqual(float(classes[0]["recall"]), 2 / 3)
        self.assertAlmostEqual(float(classes[0]["f1"]), 0.8)
        self.assertAlmostEqual(float(classes[1]["precision"]), 0.5)
        self.assertAlmostEqual(float(classes[1]["f1"]), 2 / 3)
        self.assertEqual(float(classes[2]["f1"]), 0.0)
        for filename in ("loss.png", "f1.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "per_class.png"):
            data = (self.root / "graphs" / filename).read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"), filename)
            self.assertGreater(len(data), 1000, filename)
        self.assertEqual(pyplot.get_fignums(), initial_figures)
        self.assertFalse((self.root / "graphs" / "validation_loss.png").exists())

    def test_row_normalization_handles_absent_classes_without_nan(self):
        matrix = np.asarray(self.result["metrics"]["confusion_matrix"])
        normalized = self.reporter._normalized_confusion(matrix)
        np.testing.assert_allclose(normalized[0], [2 / 3, 1 / 3, 0])
        np.testing.assert_allclose(normalized[1], [0, 1, 0])
        np.testing.assert_array_equal(normalized[2], [0, 0, 0])
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_search_and_validation_curves_are_separate_from_final_training(self):
        search = {
            "best_trial": 2,
            "trials": [
                {"number": 0, "state": "COMPLETE", "mean_validation_f1_macro": 0.6},
                {"number": 1, "state": "RUNNING", "mean_validation_f1_macro": None},
                {"number": 2, "state": "COMPLETE", "mean_validation_f1_macro": 0.8},
            ],
        }
        validation = [{**row, "validation_loss": 0.9, "validation_f1_macro": 0.4} for row in self.history]
        captured = {}

        def save(figure, name):
            captured[name] = {
                "title": figure.axes[0].get_title(),
                "lines": [line.get_label() for line in figure.axes[0].lines],
            }
            figure.clear()

        with patch.object(self.reporter, "_save", side_effect=save):
            self.reporter.write(self.history, self.result, search, [validation])
        self.assertIn("final", captured["loss.png"]["title"])
        self.assertNotIn("Validação", captured["loss.png"]["lines"])
        self.assertEqual(captured["validation_loss.png"]["lines"], ["Treino", "Validação"])
        self.assertEqual(captured["validation_f1_macro.png"]["lines"], ["Treino", "Validação"])
        self.assertIn("search.png", captured)

    def test_misaligned_predictions_fail_before_writing_tables(self):
        self.result["sample_ids"].pop()
        with self.assertRaisesRegex(ValueError, "mesmo tamanho"):
            self.reporter.write(self.history, self.result)
        self.assertFalse((self.root / "predictions.csv").exists())

    def test_rejects_logits_mislabeled_as_probabilities(self):
        self.result["metrics"]["probabilities"][0] = [0.8, 0.8, 0.8]
        with self.assertRaisesRegex(ValueError, "somar 1"):
            self.reporter.write(self.history, self.result)


if __name__ == "__main__":
    unittest.main()
