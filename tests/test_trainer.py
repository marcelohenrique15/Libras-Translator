import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from training.trainer import Trainer


class TinyDataset(Dataset):
    def __init__(self, sample_count=6, augmentation=False):
        self.rows = [{"sample_id": f"sample_{index}"} for index in range(sample_count)]
        self.augmentation = augmentation
        self.visited = []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.visited.append(index)
        label = index % 2
        inputs = torch.tensor([index / 10, label, 1 - label], dtype=torch.float32)
        if self.augmentation:
            # Os três geradores precisam ser restaurados após uma interrupção.
            inputs[0] += random.random() * 0.05
            inputs[1] += np.random.random() * 0.05
            inputs[2] += torch.rand(()) * 0.05
        return inputs, label


class RecordingClassifier(nn.Module):
    def __init__(self, batchnorm=False):
        super().__init__()
        layers = [nn.Linear(3, 4)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(4))
        layers.extend([nn.ReLU(), nn.Linear(4, 2)])
        self.network = nn.Sequential(*layers)
        self.batch_sizes = []

    def forward(self, inputs):
        self.batch_sizes.append(len(inputs))
        return self.network(inputs)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)
        self.mapping = {"class_0": 0, "class_1": 1}
        self.config = {
            "device": "cpu",
            "epochs": 3,
            "batch_size": 3,
            "learning_rate": 0.02,
            "weight_decay": 0.001,
            "patience": 10,
            "num_workers": 0,
        }

    def _seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)

    def _mlp(self):
        return nn.Sequential(
            nn.Linear(3, 6), nn.ReLU(), nn.Dropout(0.4), nn.Linear(6, 2),
        )

    def test_selection_and_early_stopping_use_f1_and_test_loads_best(self):
        config = {**self.config, "epochs": 10, "patience": 2}
        trainer = Trainer(self.root / "selection", self.mapping, config)
        model = nn.Linear(3, 2)
        training = TinyDataset()
        validation = TinyDataset(4)
        testing = TinyDataset(2)
        validation_metrics = [
            {"loss": 0.9, "accuracy": 0.90, "f1_macro": 0.40},
            {"loss": 0.8, "accuracy": 0.80, "f1_macro": 0.60},
            {"loss": 0.7, "accuracy": 0.95, "f1_macro": 0.50},
            {"loss": 0.6, "accuracy": 0.97, "f1_macro": 0.60},
        ]
        epochs = []

        def train_epoch(current_model, optimizer, dataset):
            self.assertIs(dataset, training)
            epochs.append(len(epochs) + 1)
            with torch.no_grad():
                current_model.weight.fill_(epochs[-1])
            return {"loss": 1.0, "f1_macro": 0.2}

        def evaluate(current_model, dataset):
            if dataset is validation:
                return validation_metrics.pop(0)
            self.assertIs(dataset, testing)
            # A última época tem peso 4; o melhor checkpoint tem peso 2.
            self.assertTrue(torch.all(current_model.weight == 2))
            return {"loss": 0.8, "accuracy": 0.8, "f1_macro": 0.6}

        with patch.object(trainer, "_train_epoch", side_effect=train_epoch):
            with patch.object(trainer, "_evaluate", side_effect=evaluate) as evaluation:
                result = trainer.train(model, training, validation)
                self.assertEqual(evaluation.call_count, 4)
                self.assertTrue(all(call.args[1] is validation for call in evaluation.call_args_list))
                self.assertFalse((trainer.output_dir / "test.json").exists())
                with torch.no_grad():
                    model.weight.fill_(99)
                test_result = trainer.test(model, testing)
                self.assertEqual(evaluation.call_count, 5)

        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["epochs_trained"], 4)
        self.assertEqual(result["validation_f1_macro"], 0.6)
        self.assertEqual(test_result["sample_ids"], [row["sample_id"] for row in testing.rows])
        history = json.loads((trainer.output_dir / "history.json").read_text())
        self.assertEqual([epoch["validation_f1_macro"] for epoch in history], [0.4, 0.6, 0.5, 0.6])
        last = torch.load(trainer.output_dir / "last.pt", weights_only=True)
        self.assertEqual(last["stale_epochs"], 2)

    def test_resuming_restores_model_optimizer_and_all_random_generators(self):
        self._seed(123)
        continuous = Trainer(self.root / "continuous", self.mapping, self.config)
        continuous_result = continuous.train(self._mlp(), TinyDataset(augmentation=True), TinyDataset(4))
        expected_next_random = (random.random(), float(np.random.random()), torch.rand(3))
        continuous_last = torch.load(continuous.output_dir / "last.pt", weights_only=True)

        self._seed(123)
        interrupted = Trainer(self.root / "interrupted", self.mapping, self.config)
        original_train_epoch = interrupted._train_epoch
        calls = 0

        def stop_before_second_epoch(model, optimizer, dataset):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Interrupção simulada antes da segunda época.")
            return original_train_epoch(model, optimizer, dataset)

        with patch.object(interrupted, "_train_epoch", side_effect=stop_before_second_epoch):
            with self.assertRaisesRegex(RuntimeError, "Interrupção simulada"):
                interrupted.train(self._mlp(), TinyDataset(augmentation=True), TinyDataset(4))
        first_epoch = torch.load(interrupted.output_dir / "last.pt", weights_only=True)
        self.assertEqual(first_epoch["epoch"], 1)
        self.assertTrue(first_epoch["optimizer_state_dict"]["state"])
        self.assertFalse((interrupted.output_dir / "validation.json").exists())

        # Uma sessão nova pode começar com pesos e estados aleatórios diferentes.
        self._seed(999)
        resumed = Trainer(interrupted.output_dir, self.mapping, self.config)
        resumed_result = resumed.train(self._mlp(), TinyDataset(augmentation=True), TinyDataset(4))
        actual_next_random = (random.random(), float(np.random.random()), torch.rand(3))
        resumed_last = torch.load(resumed.output_dir / "last.pt", weights_only=True)

        self.assertEqual(continuous_result, resumed_result)
        self.assertEqual(continuous_last["history"], resumed_last["history"])
        self.assertEqual(resumed_last["epoch"], 3)
        for name, expected in continuous_last["model_state_dict"].items():
            self.assertTrue(torch.equal(expected, resumed_last["model_state_dict"][name]), name)
        expected_optimizer = continuous_last["optimizer_state_dict"]
        actual_optimizer = resumed_last["optimizer_state_dict"]
        self.assertEqual(expected_optimizer["param_groups"], actual_optimizer["param_groups"])
        for parameter_id, expected_state in expected_optimizer["state"].items():
            for name, expected in expected_state.items():
                self.assertTrue(torch.equal(expected, actual_optimizer["state"][parameter_id][name]), name)
        self.assertEqual(expected_next_random[:2], actual_next_random[:2])
        self.assertTrue(torch.equal(expected_next_random[2], actual_next_random[2]))

    def test_batch_size_one_is_supported_without_batchnorm(self):
        config = {**self.config, "batch_size": 1}
        trainer = Trainer(self.root / "single", self.mapping, config)
        model = RecordingClassifier()
        dataset = TinyDataset(5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        metrics = trainer._train_epoch(model, optimizer, dataset)
        self.assertEqual(model.batch_sizes, [1] * 5)
        self.assertEqual(sorted(dataset.visited), list(range(5)))
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 5)
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_batchnorm_merges_singleton_without_losing_samples(self):
        config = {**self.config, "batch_size": 4}
        trainer = Trainer(self.root / "batchnorm", self.mapping, config)
        model = RecordingClassifier(batchnorm=True)
        dataset = TinyDataset(9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        metrics = trainer._train_epoch(model, optimizer, dataset)
        self.assertEqual(model.batch_sizes, [4, 5])
        self.assertEqual(sorted(dataset.visited), list(range(9)))
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 9)
        with self.assertRaisesRegex(ValueError, "2 amostras"):
            trainer._train_batches(1, minimum_batch=2)

    def test_loader_with_one_worker_trains_all_samples(self):
        config = {**self.config, "batch_size": 2, "num_workers": 1}
        trainer = Trainer(self.root / "workers", self.mapping, config)
        model = RecordingClassifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        metrics = trainer._train_epoch(model, optimizer, TinyDataset(4))
        self.assertEqual(model.batch_sizes, [2, 2])
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 4)
        self.assertTrue(np.isfinite(metrics["loss"]))


if __name__ == "__main__":
    unittest.main()
