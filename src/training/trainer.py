import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, output_dir: Path, class_to_index: dict[str, int], config: dict) -> None:
        self.output_dir = output_dir
        self.class_to_index = class_to_index
        self.config = config
        self.device = torch.device(config["device"])
        self.criterion = nn.CrossEntropyLoss()

    # Público
    def train(self, model: nn.Module, train_dataset, validation_dataset) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        validation_path = self.output_dir / "validation.json"
        if validation_path.exists() and (self.output_dir / "best.pt").exists():
            return json.loads(validation_path.read_text(encoding="utf-8"))

        model.to(self.device)
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        start_epoch, best_f1, stale_epochs, history = 1, -1.0, 0, []
        last_path = self.output_dir / "last.pt"
        if last_path.exists():
            last = torch.load(last_path, map_location=self.device, weights_only=True)
            model.load_state_dict(last["model_state_dict"])
            optimizer.load_state_dict(last["optimizer_state_dict"])
            start_epoch = last["epoch"] + 1
            best_f1, stale_epochs, history = last["best_f1"], last["stale_epochs"], last["history"]
            self._restore_random_state(last["random_state"])
            print(f"Retomando a partir da época {start_epoch}.", flush=True)

        for epoch in range(start_epoch, self.config["epochs"] + 1):
            if stale_epochs >= self.config["patience"]:
                break
            training = self._train_epoch(model, optimizer, train_dataset)
            validation = self._evaluate(model, validation_dataset)
            history.append({
                "epoch": epoch,
                "train_loss": training["loss"],
                "train_f1_macro": training["f1_macro"],
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_f1_macro": validation["f1_macro"],
            })
            print(
                f"Época {epoch}/{self.config['epochs']} | "
                f"treino F1={training['f1_macro']:.4f}, loss={training['loss']:.4f} | "
                f"val F1={validation['f1_macro']:.4f}, loss={validation['loss']:.4f}", flush=True,
            )

            if validation["f1_macro"] > best_f1:
                best_f1, stale_epochs = validation["f1_macro"], 0
                self._save_torch("best.pt", {
                    "model_state_dict": model.state_dict(), "class_to_index": self.class_to_index,
                    "config": self.config, "epoch": epoch,
                    "validation_accuracy": validation["accuracy"], "validation_f1_macro": best_f1,
                })
            else:
                stale_epochs += 1

            self._save_torch("last.pt", {
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "best_f1": best_f1, "stale_epochs": stale_epochs,
                "history": history, "random_state": self._random_state(),
            })
            self.write_json("history.json", history)

        best = torch.load(self.output_dir / "best.pt", map_location="cpu", weights_only=True)
        result = {
            "best_epoch": best["epoch"], "validation_accuracy": best["validation_accuracy"],
            "validation_f1_macro": best["validation_f1_macro"], "epochs_trained": len(history),
        }
        self.write_json("history.json", history)
        self.write_json("validation.json", result)
        return result

    def test(self, model: nn.Module, dataset) -> dict:
        test_path = self.output_dir / "test.json"
        if test_path.exists():
            return json.loads(test_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(self.output_dir / "best.pt", map_location=self.device, weights_only=True)
        model.to(self.device).load_state_dict(checkpoint["model_state_dict"])
        result = {"metrics": self._evaluate(model, dataset), "sample_ids": [row["sample_id"] for row in dataset.rows]}
        self.write_json("test.json", result)
        print(f"Teste: F1-macro={result['metrics']['f1_macro']:.4f}", flush=True)
        return result

    def write_json(self, name: str, data) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=self.output_dir, suffix=".tmp", delete=False, encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            temporary_path = Path(file.name)
        temporary_path.replace(self.output_dir / name)

    # Privado
    def _train_epoch(self, model, optimizer, dataset) -> dict:
        model.train()
        batchnorm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        minimum_batch = 2 if any(isinstance(layer, batchnorm) and layer.training for layer in model.modules()) else 1
        batches = self._train_batches(len(dataset), minimum_batch)
        loader = self._loader(dataset, batches=batches)
        total_loss, expected, predicted = 0.0, [], []
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            expected.extend(labels.cpu().tolist())
            predicted.extend(logits.detach().argmax(dim=1).cpu().tolist())
        return {"loss": total_loss / len(dataset), **self._metrics(expected, predicted)}

    def _train_batches(self, sample_count: int, minimum_batch: int = 1) -> list[list[int]]:
        batch_size = self.config["batch_size"]
        if sample_count < minimum_batch or batch_size < minimum_batch:
            raise ValueError(f"Este modelo precisa de pelo menos {minimum_batch} amostras por lote de treino.")
        indices = torch.randperm(sample_count).tolist()
        batches = [indices[start:start + batch_size] for start in range(0, sample_count, batch_size)]
        if minimum_batch == 2 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        return batches

    def _loader(self, dataset, batches=None) -> DataLoader:
        options = {"num_workers": self.config.get("num_workers", 0), "pin_memory": self.device.type == "cuda"}
        if options["num_workers"] > 0:
            options["multiprocessing_context"] = "spawn"
        if batches is not None:
            return DataLoader(dataset, batch_sampler=batches, **options)
        return DataLoader(dataset, batch_size=self.config["batch_size"], **options)

    def _evaluate(self, model, dataset) -> dict:
        model.eval()
        total_loss, expected, predicted = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in self._loader(dataset):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = model(inputs)
                total_loss += self.criterion(logits, labels).item() * len(labels)
                expected.extend(labels.cpu().tolist())
                predicted.extend(logits.argmax(dim=1).cpu().tolist())
        return {"loss": total_loss / len(dataset), **self._metrics(expected, predicted), "expected": expected, "predicted": predicted}

    def _metrics(self, expected: list[int], predicted: list[int]) -> dict:
        matrix = np.zeros((len(self.class_to_index), len(self.class_to_index)), dtype=np.int64)
        np.add.at(matrix, (expected, predicted), 1)
        true_positive = matrix.diagonal()
        precision = true_positive / np.maximum(matrix.sum(axis=0), 1)
        recall = true_positive / np.maximum(matrix.sum(axis=1), 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        return {
            "accuracy": float(true_positive.sum() / matrix.sum()), "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()), "f1_macro": float(f1.mean()), "confusion_matrix": matrix.tolist(),
        }

    def _save_torch(self, name: str, data: dict) -> None:
        with tempfile.NamedTemporaryFile(dir=self.output_dir, suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            torch.save(data, file)
        temporary_path.replace(self.output_dir / name)

    def _random_state(self) -> dict:
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        }

    def _restore_random_state(self, state: dict) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())
        numpy_state = state["numpy"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        if state["cuda"]:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])
