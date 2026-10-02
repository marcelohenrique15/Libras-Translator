import csv
import hashlib
import inspect
import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
import torchvision

from dataset.manifest_builder import ManifestBuilder
from dataset.sign_dataset import SignDataset
from models.model_registry import ModelRegistry
from preprocessor.landmark_processor import LandmarkProcessor
from preprocessor.processor_registry import ProcessorRegistry
from training.config import TrainingConfig
from training.hyperparameter_search import HyperparameterSearch
from training.trainer import Trainer


class TrainingPipeline():
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.dataset_root = config.dataset
        self.manifest_path = config.dataset / "metadata" / "manifest.csv"
        self.runs_dir = config.output_dir / config.dataset.resolve().name
        self.device = config.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA não está disponível neste ambiente; use --device cpu.")

        processor_options = dict(config.processor_options)
        processor_class = ProcessorRegistry.get_class(config.processor)
        self.uses_landmarks = issubclass(processor_class, LandmarkProcessor)
        if self.uses_landmarks:
            representation = ModelRegistry.input_format(config.model)
            processor_options.setdefault("representation", representation)
            if processor_options["representation"] != representation:
                raise ValueError(f"O modelo {config.model} recebe {representation}; ajuste processor_options.representation.")
        self.processor = ProcessorRegistry.create(config.processor, config.dataset, processor_options)
        self.processor_options = processor_options

    # Público
    def prepare(self, stage: str = "encode") -> list[dict[str, str]]:
        print("1. Manifesto", flush=True)
        rows = self._prepare_manifest()
        if self.uses_landmarks:
            from preprocessor.landmarks_extractor import LandmarksExtractor

            print("2. Extração de landmarks", flush=True)
            LandmarksExtractor(self.dataset_root).extract()
        print(f"3. Pré-processamento até a etapa: {stage}", flush=True)
        self.processor.prepare_all(rows, stage=stage)
        return rows

    def train(self) -> Path:
        if self.config.search_config is not None:
            return self._search()

        rows = self.prepare()
        results = []
        for test_signer in self._test_signers(rows):
            print(f"4. Treino e validação | teste reservado: {test_signer}", flush=True)
            folds = self._train_folds(rows, test_signer)
            print("5. Avaliação dos checkpoints no teste", flush=True)
            results.extend(self._test_folds(rows, test_signer, folds))

        run_dir = self._run_dir(rows)
        name = "summary_all.json" if self.config.test_signer_id == "all" else f"summary_{self._test_signers(rows)[0]}.json"
        if self.config.validation_signer_id is not None:
            name = name.replace(".json", f"_val_{self.config.validation_signer_id}.json")
        self._writer(run_dir, rows).write_json(name, self._summary(results))
        print(f"Resumo salvo em: {run_dir / name}", flush=True)
        return run_dir / name

    # Privado
    def _prepare_manifest(self) -> list[dict[str, str]]:
        if not self.manifest_path.exists():
            ManifestBuilder(self.dataset_root, self.config.dataset_format).build()
            print(f"Manifesto criado: {self.manifest_path}", flush=True)
        else:
            print(f"Reutilizando manifesto: {self.manifest_path}", flush=True)
        with self.manifest_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError("O manifesto está vazio.")
        return sorted(rows, key=lambda row: row["sample_id"])

    def _test_signers(self, rows: list[dict[str, str]]) -> list[str]:
        signers = sorted({row["signer_id"] for row in rows})
        if len(signers) < 3:
            raise ValueError("Treino, validação e teste por sinalizador precisam de pelo menos três sinalizadores.")
        if self.config.test_signer_id == "all":
            return signers
        if self.config.test_signer_id is not None:
            return [self._signer_id(self.config.test_signer_id, signers)]
        current_test = sorted({row["signer_id"] for row in rows if row.get("set") == "test"})
        if len(current_test) > 1:
            raise ValueError("Escolha --test-signer-id: o manifesto contém mais de um sinalizador de teste.")
        return current_test or [signers[0]]

    def _validation_signers(self, rows, test_signer: str) -> list[str]:
        signers = sorted({row["signer_id"] for row in rows} - {test_signer})
        if self.config.validation_signer_id is not None:
            return [self._signer_id(self.config.validation_signer_id, signers)]
        return signers

    def _signer_id(self, value: str, signers: list[str]) -> str:
        if value in signers:
            return value
        if value.zfill(2) in signers:
            return value.zfill(2)
        raise ValueError(f"Sinalizador ausente nesta divisão: {value}")

    def _model_options(self) -> dict:
        model_class = ModelRegistry.get_class(self.config.model)
        parameters = dict(self.config.model_options)
        signature = inspect.signature(model_class)
        if "input_size" in signature.parameters and hasattr(self.processor, "input_size"):
            parameters.setdefault("input_size", self.processor.input_size)
        for name, argument in signature.parameters.items():
            if name != "class_count" and argument.default is not inspect.Parameter.empty:
                parameters.setdefault(name, argument.default)
        return parameters

    def _settings(self) -> dict:
        settings = self.config.to_dict()
        for name in ("dataset", "output_dir", "weights_dir", "test_signer_id", "validation_signer_id", "search_config", "search_trials", "num_workers"):
            settings.pop(name)
        return {
            **settings, "device": self.device, "model_options": self._model_options(),
            "processor_options": self.processor_options, "loss": "cross_entropy", "selection_metric": "f1_macro",
            "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        }

    def _run_id(self, rows) -> str:
        signature = hashlib.sha256(json.dumps(self._settings(), sort_keys=True).encode())
        source_paths = [
            Path(__file__), Path(inspect.getfile(Trainer)), Path(inspect.getfile(SignDataset)),
            ModelRegistry.source_path(self.config.model), ProcessorRegistry.source_path(self.config.processor),
        ]
        for path in source_paths:
            signature.update(path.read_bytes())
        for row in rows:
            metadata = {key: value for key, value in row.items() if key != "set"}
            signature.update(json.dumps(metadata, sort_keys=True).encode())
            if self.uses_landmarks:
                source = self.dataset_root / "processed" / "landmarks" / f"{row['sample_id']}.csv"
            else:
                source = self.dataset_root / row["path"]
            stat = source.stat()
            signature.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        return signature.hexdigest()[:12]

    def _run_dir(self, rows) -> Path:
        return self.runs_dir / self._run_id(rows)

    def _class_mapping(self, rows) -> dict[str, int]:
        return {class_id: index for index, class_id in enumerate(sorted({row["class_id"] for row in rows}))}

    def _writer(self, output_dir: Path, rows) -> Trainer:
        config = {**self._settings(), "num_workers": self.config.num_workers}
        return Trainer(output_dir, self._class_mapping(rows), config)

    def _create_model(self, rows, load_checkpoint: bool = False):
        self.config.weights_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(self.config.weights_dir))
        parameters = self._model_options()
        if load_checkpoint and "pretrained" in parameters:
            parameters["pretrained"] = False
        return ModelRegistry.create(self.config.model, len(self._class_mapping(rows)), parameters)

    def _seed(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.device == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 8))

    def _train_folds(self, rows, test_signer: str) -> list[dict]:
        run_dir = self._run_dir(rows)
        self._writer(run_dir, rows).write_json("config.json", self._settings())
        mapping = self._class_mapping(rows)
        folds = []
        for validation_signer in self._validation_signers(rows, test_signer):
            fold_dir = run_dir / f"test_{test_signer}" / f"val_{validation_signer}"
            trainer = self._writer(fold_dir, rows)
            validation_path = fold_dir / "validation.json"
            if validation_path.exists() and (fold_dir / "best.pt").exists():
                print(f"Reutilizando treino: teste {test_signer}, validação {validation_signer}.", flush=True)
                result = json.loads(validation_path.read_text(encoding="utf-8"))
            else:
                print(f"Treinando {self.config.model} | teste {test_signer}, validação {validation_signer}", flush=True)
                self._seed()
                train_rows = [row for row in rows if row["signer_id"] not in {test_signer, validation_signer}]
                validation_rows = [row for row in rows if row["signer_id"] == validation_signer]
                train_dataset = SignDataset(train_rows, self.processor, mapping, training=True)
                validation_dataset = SignDataset(validation_rows, self.processor, mapping)
                trainer.config.update({"test_signer": test_signer, "validation_signer": validation_signer})
                model = self._create_model(rows, load_checkpoint=(fold_dir / "last.pt").exists())
                result = trainer.train(model, train_dataset, validation_dataset)
            folds.append({"test_signer": test_signer, "validation_signer": validation_signer, **result})
        return folds

    def _test_folds(self, rows, test_signer: str, folds: list[dict]) -> list[dict]:
        test_rows = [row for row in rows if row["signer_id"] == test_signer]
        dataset = SignDataset(test_rows, self.processor, self._class_mapping(rows))
        results = []
        for fold in folds:
            fold_dir = self._run_dir(rows) / f"test_{test_signer}" / f"val_{fold['validation_signer']}"
            trainer = self._writer(fold_dir, rows)
            if (fold_dir / "test.json").exists():
                print(f"Reutilizando teste: {test_signer}, validação {fold['validation_signer']}.", flush=True)
                test = json.loads((fold_dir / "test.json").read_text(encoding="utf-8"))
            else:
                test = trainer.test(self._create_model(rows, load_checkpoint=True), dataset)
            result = {**fold, "test": test["metrics"], "test_samples": test["sample_ids"]}
            trainer.write_json("result.json", result)
            results.append(result)
        return results

    def _search(self) -> Path:
        search = HyperparameterSearch(self.config.search_config)
        rows = self.prepare()
        test_signers = self._test_signers(rows)
        divisions = {
            signer: self._validation_signers(rows, signer)
            for signer in test_signers
        }
        # O orçamento fica fora da assinatura para ampliar o mesmo estudo depois.
        search_signature = {
            "base_run": self._run_id(rows), "parameters": search.parameters,
            "divisions": divisions, "sampler": "TPE",
            "n_startup_trials": search.n_startup_trials, "optuna_version": optuna.__version__,
        }
        signature = hashlib.sha256(json.dumps(search_signature, sort_keys=True).encode())
        signature.update(Path(inspect.getfile(HyperparameterSearch)).read_bytes())
        search_dir = self.runs_dir / f"search_{signature.hexdigest()[:12]}"
        writer = self._writer(search_dir, rows)
        writer.write_json("search_config.json", {"base_config": self.config.to_dict(), **search_signature})
        all_results = []

        for test_signer in test_signers:
            print(f"Busca Optuna | teste reservado: {test_signer}", flush=True)
            study = search.create_study(search_dir, f"test_{test_signer}", self.config.seed)

            def evaluate(candidate: TrainingConfig, number: int) -> float:
                pipeline = TrainingPipeline(candidate)
                candidate_rows = pipeline.prepare()
                folds = pipeline._train_folds(candidate_rows, test_signer)
                score = float(np.mean([fold["validation_f1_macro"] for fold in folds]))
                writer.write_json(f"trial_test{test_signer}_{number}.json", {
                    "number": number, "run_id": pipeline._run_id(candidate_rows),
                    "config": candidate.to_dict(), "mean_validation_f1_macro": score, "folds": folds,
                })
                return score

            winner = search.run(study, self.config, evaluate, n_trials=self.config.search_trials)
            writer.write_json(f"search_test{test_signer}.json", search.report(study))
            candidate = search.configuration(self.config, winner.params)
            pipeline = TrainingPipeline(candidate)
            candidate_rows = pipeline.prepare()
            folds = pipeline._train_folds(candidate_rows, test_signer)
            best_config = candidate.with_overrides({"test_signer_id": test_signer}).to_dict()
            writer.write_json(f"best_config_test{test_signer}.json", best_config)
            print(f"Vencedor: tentativa {winner.number + 1} | F1 validação={winner.value:.4f}", flush=True)
            all_results.extend(pipeline._test_folds(candidate_rows, test_signer, folds))

        path = search_dir / "summary.json"
        writer.write_json(path.name, self._summary(all_results))
        print(f"Busca e avaliação concluídas: {path}", flush=True)
        return path

    def _summary(self, results: list[dict]) -> dict:
        summary = {"sessions": len(results), "folds": results, "metrics": {}}
        for name in ("f1_macro", "accuracy", "precision_macro", "recall_macro"):
            values = [result["test"][name] for result in results]
            summary["metrics"][name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        return summary
