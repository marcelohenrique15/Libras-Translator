"""Carregamento de pesos e classificação com o mesmo encoder do treino."""

from __future__ import annotations

import csv
import inspect
from pathlib import Path

import numpy as np

from application.config import RuntimeConfig
from application.types import Candidate


class CheckpointClassifier:
    """Carrega configuração e pesos treinados, sem downloads ou caches de dados."""

    def __init__(self, config: RuntimeConfig) -> None:
        import torch

        from models.model_registry import ModelRegistry
        from preprocessor.streaming_processor import StreamingLandmarkProcessor

        if config.checkpoint is None:
            raise ValueError("Selecione um checkpoint treinado para iniciar a aplicação.")
        path = Path(config.checkpoint)
        self.device = (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ) if config.device == "auto" else config.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA não está disponível. Inicie a aplicação com --device cpu.")
        if self.device == "cpu":
            torch.set_num_threads(config.cpu_threads)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("O arquivo não contém um checkpoint de treinamento válido.")
        training_config = checkpoint.get("config", {})
        if not isinstance(training_config, dict):
            raise ValueError("O checkpoint não contém uma configuração de treinamento válida.")
        if training_config.get("processor") != "landmarks":
            raise ValueError("A aplicação precisa de um checkpoint com processor='landmarks'.")
        mapping = checkpoint.get("class_to_index", {})
        if (
            not isinstance(mapping, dict) or len(mapping) < 2
            or any(not isinstance(key, str) for key in mapping)
            or any(type(index) is not int for index in mapping.values())
            or sorted(mapping.values()) != list(range(len(mapping)))
        ):
            raise ValueError("O checkpoint não contém um class_to_index válido e contínuo.")
        self._class_ids = tuple(sorted(mapping, key=mapping.get))
        self._labels = self._read_labels(path.parent / "per_class.csv", mapping)
        self.processor = StreamingLandmarkProcessor(
            dataset_root=path.parent, **training_config.get("processor_options", {}),
        )
        model_name = training_config.get("model", "")
        expected_format = ModelRegistry.input_format(model_name)
        if self.processor.representation != expected_format:
            raise ValueError(
                f"O modelo recebe '{expected_format}', mas o checkpoint prepara "
                f"'{self.processor.representation}'.",
            )
        options = dict(training_config.get("model_options", {}))
        if "pretrained" in inspect.signature(ModelRegistry.get_class(model_name)).parameters:
            options["pretrained"] = False
        self._model = ModelRegistry.create(model_name, len(mapping), options)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self.device).eval().requires_grad_(False)

    def predict(self, window: np.ndarray) -> tuple[Candidate, ...]:
        import torch

        prepared = self.processor.prepare_array(window)
        encoded = np.ascontiguousarray(self.processor.encode(prepared), dtype=np.float32)
        if not np.isfinite(encoded).all():
            raise ValueError("A entrada preparada contém coordenadas inválidas.")
        inputs = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self._model(inputs)
            if getattr(self._model, "OUTPUT_FORMAT", "logits") == "logits":
                output = output.softmax(dim=1)
            probabilities = output.float().cpu().numpy()
        if probabilities.shape != (1, len(self._class_ids)):
            raise ValueError("A saída do modelo não corresponde às classes do checkpoint.")
        probabilities = probabilities[0]
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isclose(probabilities.sum(), 1, atol=1e-4)
        ):
            raise ValueError("O modelo retornou probabilidades inválidas.")
        indices = np.argsort(-probabilities, kind="stable")[:3]
        return tuple(
            Candidate(self._class_ids[index], self._labels[index], float(probabilities[index]))
            for index in indices
        )

    @staticmethod
    def _read_labels(path: Path, mapping: dict[str, int]) -> tuple[str, ...]:
        labels = [class_id for class_id in sorted(mapping, key=mapping.get)]
        if not path.exists():
            return tuple(labels)
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != len(mapping):
            raise ValueError("per_class.csv não contém as mesmas classes do checkpoint.")
        seen = set()
        for row in rows:
            try:
                index = int(row["class_index"])
                class_id, separator, label = row["class_label"].partition(" · ")
                if (
                    not separator or not label.strip() or class_id not in mapping
                    or mapping[class_id] != index or index in seen
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Os rótulos de per_class.csv não correspondem à ordem do checkpoint.",
                ) from error
            labels[index] = label.strip()
            seen.add(index)
        return tuple(labels)
