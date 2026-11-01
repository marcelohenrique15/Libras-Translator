import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from application.config import RuntimeConfig, resolve_checkpoint
from preprocessor.streaming_processor import StreamingLandmarkProcessor


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_config_reads_auto_and_rejects_invalid_controls(self):
        path = self.root / "app.toml"
        path.write_text('checkpoint = "auto"\nwindow_seconds = 2.0\nmirror = false\nwindow_width = 1440\nwindow_height = 810\n')
        config = RuntimeConfig.from_toml(path)
        self.assertIsNone(config.checkpoint)
        self.assertEqual(config.window_seconds, 2)
        self.assertFalse(config.mirror)
        self.assertEqual((config.window_width, config.window_height), (1440, 810))
        for values in ({"width": 0}, {"stability_count": 1.5}, {"cpu_threads": 0},
                       {"prediction_interval": float("nan")}, {"window_seconds": -1},
                       {"confidence_threshold": 1.1}, {"confidence_threshold": "alta"},
                       {"confidence_threshold": True}, {"sample_fps": float("inf")},
                       {"window_seconds": 0.01, "sample_fps": 1},
                       {"window_width": 0}, {"window_height": 1.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RuntimeConfig(**values)

    def test_latest_completed_checkpoint_is_selected_by_time_not_test_score(self):
        for name, timestamp, score in (("old", 100, 1.0), ("new", 200, 0.5)):
            folder = self.root / name
            folder.mkdir()
            result = folder / "result.json"
            result.write_text(json.dumps({"test": {"accuracy": score}}))
            (folder / "final.pt").touch()
            os.utime(result, (timestamp, timestamp))
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        (incomplete / "result.json").write_text("{}")
        config = resolve_checkpoint(RuntimeConfig(), self.root)
        self.assertEqual(config.checkpoint, self.root / "new/final.pt")
        explicit = resolve_checkpoint(RuntimeConfig(checkpoint=self.root / "old/final.pt"), self.root)
        self.assertEqual(explicit.checkpoint, self.root / "old/final.pt")

    def test_no_checkpoint_is_a_clear_startup_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Nenhum modelo final"):
            resolve_checkpoint(RuntimeConfig(), self.root)

    def test_live_preparation_matches_saved_training_pipeline_without_mutating_input(self):
        for imputation in (False, True):
            with self.subTest(imputation=imputation):
                processor = StreamingLandmarkProcessor(self.root, imputation=imputation)
                points = np.full((9, processor.landmark_count, 2), 0.45, dtype=np.float32)
                points[:, 18:39, 0] += np.arange(9, dtype=np.float32)[:, None] / 100
                points[3, 20] = np.nan
                before = points.copy()
                columns = [f"{group}_{point}_{axis}" for group, indices in processor.landmark_groups
                           for point in indices for axis in ("x", "y")]
                processor.landmarks_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(points.reshape(len(points), -1), columns=columns).to_csv(
                    processor.landmarks_dir / "sample.csv", index=False,
                )
                prepared_live = processor.prepare_array(points)
                np.testing.assert_array_equal(points, before)
                np.testing.assert_allclose(prepared_live, processor.prepare("sample"), atol=1e-6, equal_nan=True)
                np.testing.assert_array_equal(processor.encode(prepared_live), processor.process("sample"))

    def test_live_preparation_checks_shape_without_creating_caches(self):
        processor = StreamingLandmarkProcessor(self.root)
        for shape in ((0, 80, 2), (3, 79, 2), (3, 80, 3), (3, 160)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "formato"):
                processor.prepare_array(np.zeros(shape))
        self.assertFalse((self.root / "processed").exists())
