import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from training.inference_benchmark import InferenceBenchmark
from training.reporting import ExperimentReporter
from training.trainer import Trainer


class SampleDataset:
    rows = [{"sample_id": "a"}, {"sample_id": "b"}]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return torch.tensor([float(index), 1.0]), index


class CheckedModel(nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.calls = 0

    def forward(self, inputs):
        assert not self.training and torch.is_inference_mode_enabled()
        assert inputs.shape == (1, 2)
        self.calls += 1
        return super().forward(inputs)


class InferenceBenchmarkTests(unittest.TestCase):
    def test_serial_timings_exclude_warmup_and_include_all_stages(self):
        model = CheckedModel()
        # Início da sessão; cinco fronteiras por amostra; fim da sessão.
        clock = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        with patch("training.inference_benchmark.perf_counter", side_effect=clock):
            result = InferenceBenchmark("cpu", warmup_iterations=3).run(model, SampleDataset())
        self.assertEqual(model.calls, 5)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual([row["sample_id"] for row in result["samples"]], ["a", "b"])
        self.assertEqual(result["timings"]["inference_ms"]["mean"], 1000)
        self.assertEqual(result["timings"]["processing_ms"]["p95"], 4000)
        self.assertAlmostEqual(result["throughput_samples_per_second"], 2 / 11)
        for row in result["samples"]:
            self.assertEqual(row["processing_ms"], sum(row[key] for key in (
                "data_ms", "transfer_ms", "inference_ms", "postprocess_ms")))
        json.dumps(result)

    def test_gpu_synchronization_targets_only_selected_device(self):
        with patch("torch.cuda.synchronize") as synchronize:
            InferenceBenchmark("cpu")._synchronize()
            synchronize.assert_not_called()
            InferenceBenchmark("cuda:1")._synchronize()
            synchronize.assert_called_once_with(torch.device("cuda:1"))

    def test_existing_accuracy_result_gets_benchmark_without_repeating_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = nn.Linear(2, 2)
            trainer = Trainer(root, {"a": 0, "b": 1}, {"device": "cpu"})
            original = {"metrics": {"f1_macro": 0.5}, "sample_ids": ["a", "b"]}
            trainer.write_json("test.json", original)
            torch.save({"model_state_dict": model.state_dict()}, root / "final.pt")
            with patch.object(trainer, "_evaluate", side_effect=AssertionError("Avaliação repetida")):
                result = trainer.test(model, SampleDataset())
            self.assertEqual(result["metrics"], original["metrics"])
            self.assertTrue((root / "performance.json").is_file())
            self.assertEqual(result["performance"]["sample_count"], 2)
            with patch.object(InferenceBenchmark, "run", side_effect=AssertionError("Benchmark repetido")):
                self.assertEqual(trainer.test(model, SampleDataset()), result)

    def test_report_exports_measurements_and_performance_graph(self):
        performance = InferenceBenchmark("cpu").run(nn.Linear(2, 2), SampleDataset())
        with tempfile.TemporaryDirectory() as directory:
            reporter = ExperimentReporter(Path(directory), ["a", "b"])
            (Path(directory) / "graphs").mkdir()
            reporter.write_performance(performance)
            self.assertTrue((Path(directory) / "performance_samples.csv").is_file())
            self.assertTrue((Path(directory) / "graphs/performance.png").read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
