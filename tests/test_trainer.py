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
    def __init__(self, sample_count=6, consume_random=False):
        self.rows = [{"sample_id": f"sample_{index}"} for index in range(sample_count)]
        self.consume_random = consume_random
        self.visited = []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.visited.append(index)
        label = index % 2
        inputs = torch.tensor([index / 10, label, 1 - label], dtype=torch.float32)
        if self.consume_random:
            # Os três geradores precisam ser restaurados após uma interrupção.
            # Consumir estados não altera os dados nem aplica augmentation.
            random.random()
            np.random.random()
            torch.rand(())
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


class ProbabilityClassifier(nn.Module):
    OUTPUT_FORMAT = "probabilities"

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2)

    def forward_logits(self, inputs):
        return self.layer(inputs)

    def forward(self, inputs):
        return self.forward_logits(inputs).softmax(dim=1)


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
        continuous_result = continuous.train(self._mlp(), TinyDataset(consume_random=True), TinyDataset(4))
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
                interrupted.train(self._mlp(), TinyDataset(consume_random=True), TinyDataset(4))
        first_epoch = torch.load(interrupted.output_dir / "last.pt", weights_only=True)
        self.assertEqual(first_epoch["epoch"], 1)
        self.assertTrue(first_epoch["optimizer_state_dict"]["state"])
        self.assertFalse((interrupted.output_dir / "validation.json").exists())

        # Uma sessão nova pode começar com pesos e estados aleatórios diferentes.
        self._seed(999)
        resumed = Trainer(interrupted.output_dir, self.mapping, self.config)
        resumed_result = resumed.train(self._mlp(), TinyDataset(consume_random=True), TinyDataset(4))
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

    def test_final_fit_runs_fixed_epochs_without_validation_and_test_uses_final_weights(self):
        config = {**self.config, "patience": 1}
        trainer = Trainer(self.root / "final", self.mapping, config)
        model = nn.Linear(3, 2)
        initial = model.weight.detach().clone()
        with patch.object(trainer, "_evaluate", side_effect=AssertionError("refit não valida")):
            result = trainer.fit(model, TinyDataset())
        self.assertEqual(result["epochs_trained"], 3)
        self.assertEqual(result["sample_count"], 6)
        self.assertFalse(torch.equal(initial, model.weight))
        self.assertFalse((trainer.output_dir / "best.pt").exists())
        self.assertFalse((trainer.output_dir / "validation.json").exists())
        final = torch.load(trainer.output_dir / "final.pt", weights_only=True)
        self.assertEqual(final["epoch"], 3)
        history = json.loads((trainer.output_dir / "history.json").read_text())
        self.assertEqual([row["epoch"] for row in history], [1, 2, 3])
        self.assertTrue(all("validation_loss" not in row for row in history))

        with torch.no_grad():
            model.weight.fill_(99)
        trainer._save_torch("best.pt", {"model_state_dict": model.state_dict()})
        tested = trainer.test(model, TinyDataset(4))
        self.assertTrue(torch.equal(model.weight, final["model_state_dict"]["weight"]))
        self.assertEqual(len(tested["metrics"]["probabilities"]), 4)
        for probabilities in tested["metrics"]["probabilities"]:
            self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        with patch.object(trainer, "_train_epoch", side_effect=AssertionError("refit concluído repetido")):
            self.assertEqual(trainer.fit(model, TinyDataset()), result)

    def test_final_fit_resumes_identically_after_an_interruption(self):
        self._seed(12)
        continuous = Trainer(self.root / "continuous_fit", self.mapping, self.config)
        expected = continuous.fit(self._mlp(), TinyDataset(consume_random=True))
        expected_checkpoint = torch.load(continuous.output_dir / "last.pt", weights_only=True)

        self._seed(12)
        interrupted = Trainer(self.root / "interrupted_fit", self.mapping, self.config)
        original_epoch = interrupted._train_epoch
        calls = 0

        def stop(model, optimizer, dataset):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Interrompido")
            return original_epoch(model, optimizer, dataset)

        with patch.object(interrupted, "_train_epoch", side_effect=stop):
            with self.assertRaisesRegex(RuntimeError, "Interrompido"):
                interrupted.fit(self._mlp(), TinyDataset(consume_random=True))
        self.assertFalse((interrupted.output_dir / "final.pt").exists())
        self._seed(999)
        actual = interrupted.fit(self._mlp(), TinyDataset(consume_random=True))
        actual_checkpoint = torch.load(interrupted.output_dir / "last.pt", weights_only=True)
        self.assertEqual(expected, actual)
        self.assertEqual(expected_checkpoint["history"], actual_checkpoint["history"])
        for name, weights in expected_checkpoint["model_state_dict"].items():
            self.assertTrue(torch.equal(weights, actual_checkpoint["model_state_dict"][name]), name)

    def test_final_fit_recovers_history_if_interrupted_after_last_epoch_checkpoint(self):
        trainer = Trainer(self.root / "last_epoch_crash", self.mapping, {**self.config, "epochs": 1})
        with patch.object(trainer, "write_json", side_effect=KeyboardInterrupt("Antes de salvar history.json")):
            with self.assertRaises(KeyboardInterrupt):
                trainer.fit(self._mlp(), TinyDataset())
        checkpoint = torch.load(trainer.output_dir / "last.pt", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertFalse((trainer.output_dir / "history.json").exists())
        with patch.object(trainer, "_train_epoch", side_effect=AssertionError("última época repetida")):
            result = trainer.fit(self._mlp(), TinyDataset())
        history = json.loads((trainer.output_dir / "history.json").read_text())
        self.assertEqual(history, checkpoint["history"])
        self.assertEqual(result["epochs_trained"], 1)
        self.assertTrue((trainer.output_dir / "final.pt").exists())

    def test_probability_model_uses_logits_for_cross_entropy_and_returns_test_probabilities(self):
        trainer = Trainer(self.root / "probabilities", self.mapping, {**self.config, "label_smoothing": 0.2})
        model = ProbabilityClassifier()
        inputs = torch.tensor([[0.1, 1.0, 0.0], [0.2, 0.0, 1.0]])
        labels = torch.tensor([1, 0])
        logits = trainer._forward_logits(model, inputs)
        self.assertTrue(torch.equal(logits, model.forward_logits(inputs)))
        self.assertFalse(torch.equal(logits, model(inputs)))
        logits.retain_grad()
        trainer.training_criterion(logits, labels).backward()
        expected_gradient = (logits.detach().softmax(dim=1) - torch.tensor([[0.1, 0.9], [0.9, 0.1]])) / 2
        torch.testing.assert_close(logits.grad, expected_gradient)

        dataset = TinyDataset(4)
        result = trainer._evaluate(model, dataset)
        batch_inputs = torch.stack([dataset[index][0] for index in range(len(dataset))])
        batch_labels = torch.tensor([dataset[index][1] for index in range(len(dataset))])
        expected_loss = nn.CrossEntropyLoss()(model.forward_logits(batch_inputs), batch_labels)
        self.assertAlmostEqual(result["loss"], expected_loss.item(), places=6)
        torch.testing.assert_close(torch.tensor(result["probabilities"]), model(batch_inputs))

        class InvalidProbabilityModel(nn.Linear):
            OUTPUT_FORMAT = "probabilities"

        with self.assertRaisesRegex(ValueError, "forward_logits"):
            trainer._forward_logits(InvalidProbabilityModel(3, 2), inputs)

    def test_regularization_ignores_frozen_weights_bias_and_batchnorm_and_adamw_is_decoupled(self):
        model = RecordingClassifier(batchnorm=True)
        model.network[3].weight.requires_grad_(False)
        config = {**self.config, "l1_lambda": 0.01, "l2_lambda": 0.02, "gradient_clip": 0.5}
        trainer = Trainer(self.root / "regularization", self.mapping, config)
        weights = model.network[0].weight
        expected = 0.01 * weights.abs().sum() + 0.02 * weights.square().sum()
        torch.testing.assert_close(trainer._regularization(model), expected)
        trainer._regularization(model).backward()
        torch.testing.assert_close(weights.grad, 0.01 * weights.sign() + 0.04 * weights)
        self.assertIsNone(model.network[0].bias.grad)
        self.assertIsNone(model.network[1].weight.grad)
        self.assertIsNone(model.network[3].weight.grad)
        optimizer = trainer._optimizer(model)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], config["weight_decay"])
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0)
        with patch("torch.nn.utils.clip_grad_norm_", wraps=nn.utils.clip_grad_norm_) as clip:
            result = trainer._train_epoch(model, optimizer, TinyDataset())
        self.assertEqual(clip.call_count, 2)
        self.assertGreater(result["objective"], result["loss"])

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
