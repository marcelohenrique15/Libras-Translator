import csv
import hashlib
import inspect
import json
import random
from pathlib import Path
from uuid import uuid4

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
from training.model_selection import ModelSelection
from training.reporting import ExperimentReporter
from training.trainer import Trainer


class TrainingPipeline():
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.dataset_root = config.dataset
        self.manifest_path = config.dataset / "metadata" / "manifest.csv"
        self.runs_dir = config.output_dir / config.dataset.resolve().name
        if config.force_restart:
            self.runs_dir /= f"fresh_{uuid4().hex[:8]}"
        self.device = config.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA não está disponível neste ambiente; use --device cpu.")

        processor_options = dict(config.processor_options)
        processor_class = ProcessorRegistry.get_class(config.processor)
        self.uses_landmarks = issubclass(processor_class, LandmarkProcessor)
        if self.uses_landmarks:
            processor_options.setdefault("augmentation", False)
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
        rows = self.prepare()
        search = None
        if self.config.reuse_search is not None:
            source = self.config.reuse_search.resolve()
            signature = hashlib.sha256(str(source).encode()).hexdigest()[:12]
            run_dir = self._run_dir(rows) / f"import_{signature}"
        elif self.config.search_config is not None:
            search = HyperparameterSearch(self.config.search_config)
            run_dir = self._search_dir(rows, search)
        else:
            run_dir = self._run_dir(rows)

        writer = self._writer(run_dir, rows)
        results = []
        for test_signer in self._test_signers(rows):
            print(f"4. Seleção de configuração | teste reservado: {test_signer}", flush=True)
            if self.config.reuse_search is not None:
                selection = self._import_selection(rows, test_signer)
            elif search is not None:
                selection = self._search(rows, test_signer, search, run_dir)
            elif self.config.final_epochs is not None:
                selection = ModelSelection(self.config, [], None, "provided_config")
            else:
                folds = self._train_folds(rows, test_signer)
                selection = ModelSelection(
                    self.config, folds, self._validation_score(folds), "fixed_config_validation",
                    validation_run_dir=self._run_dir(rows),
                )
            writer.write_json(f"best_config_test{test_signer}.json", selection.final_config(test_signer).to_dict())
            results.append(self._fit_and_test(rows, test_signer, selection, run_dir))

        path = run_dir / "summary.json"
        writer.write_json(path.name, self._summary(results))
        print(f"Experimento concluído: {path}", flush=True)
        return path

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
        for name in (
            "dataset", "output_dir", "weights_dir", "test_signer_id", "validation_signer_id",
            "search_config", "search_trials", "reuse_search", "force_restart", "final_epochs", "num_workers",
        ):
            settings.pop(name)
        return {
            **settings, "device": self.device, "model_options": self._model_options(),
            "processor_options": self.processor_options, "loss": "cross_entropy_logits",
            "optimizer": "adamw", "selection_metric": "f1_macro",
            "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        }

    def _run_id(self, rows) -> str:
        signature = hashlib.sha256(json.dumps(self._settings(), sort_keys=True).encode())
        signature.update(self._code_signature().encode())
        signature.update(self._dataset_signature(rows).encode())
        return signature.hexdigest()[:12]

    def _code_signature(self) -> str:
        signature = hashlib.sha256()
        source_paths = [
            Path(__file__), Path(inspect.getfile(Trainer)), Path(inspect.getfile(SignDataset)),
            Path(inspect.getfile(ModelSelection)), Path(inspect.getfile(HyperparameterSearch)),
            ModelRegistry.source_path(self.config.model), ProcessorRegistry.source_path(self.config.processor),
        ]
        for path in source_paths:
            signature.update(path.read_bytes())
        return signature.hexdigest()

    def _dataset_signature(self, rows) -> str:
        signature = hashlib.sha256()
        for row in rows:
            metadata = {key: value for key, value in row.items() if key != "set"}
            signature.update(json.dumps(metadata, sort_keys=True).encode())
            if self.uses_landmarks:
                source = self.dataset_root / "processed" / "landmarks" / f"{row['sample_id']}.csv"
            else:
                source = self.dataset_root / row["path"]
            stat = source.stat()
            signature.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        return signature.hexdigest()

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

    def _search_dir(self, rows, search: HyperparameterSearch) -> Path:
        divisions = {
            signer: self._validation_signers(rows, signer)
            for signer in self._test_signers(rows)
        }
        # O orçamento fica fora da assinatura para ampliar o mesmo estudo depois.
        search_signature = {
            "base_run": self._run_id(rows), "parameters": search.parameters,
            "dataset_signature": self._dataset_signature(rows), "protocol_version": 2,
            "code_signature": self._code_signature(),
            "divisions": divisions, "sampler": "TPE",
            "n_startup_trials": search.n_startup_trials, "optuna_version": optuna.__version__,
        }
        signature = hashlib.sha256(json.dumps(search_signature, sort_keys=True).encode())
        signature.update(Path(inspect.getfile(HyperparameterSearch)).read_bytes())
        search_dir = self.runs_dir / f"search_{signature.hexdigest()[:12]}"
        writer = self._writer(search_dir, rows)
        writer.write_json("search_config.json", {"base_config": self.config.to_dict(), **search_signature})
        return search_dir

    def _candidate_pipeline(self, config: TrainingConfig, rows) -> "TrainingPipeline":
        candidate = config.with_overrides({"force_restart": False, "reuse_search": None})
        pipeline = TrainingPipeline(candidate)
        pipeline.runs_dir = self.runs_dir
        # A extração já ocorreu. Apenas opções de processamento diferentes exigem novos caches.
        if pipeline.processor_options == self.processor_options:
            pipeline.processor = self.processor
        else:
            pipeline.processor.prepare_all(rows)
        return pipeline

    @staticmethod
    def _validation_score(folds: list[dict]) -> float:
        return float(np.mean([fold["validation_f1_macro"] for fold in folds]))

    def _search(self, rows, test_signer: str, search: HyperparameterSearch, search_dir: Path) -> ModelSelection:
        writer = self._writer(search_dir, rows)
        study = search.create_study(search_dir, f"test_{test_signer}", self.config.seed)
        validation_signers = self._validation_signers(rows, test_signer)
        print(f"Protocolo da busca | teste reservado: {test_signer} | "
              f"validação: {', '.join(validation_signers)} | {len(validation_signers)} divisão(ões) por tentativa", flush=True)

        def evaluate(candidate: TrainingConfig, number: int) -> float:
            pipeline = self._candidate_pipeline(candidate, rows)
            folds = pipeline._train_folds(rows, test_signer)
            score = self._validation_score(folds)
            writer.write_json(f"trial_test{test_signer}_{number}.json", {
                "number": number, "run_id": pipeline._run_id(rows),
                "config": candidate.to_dict(), "mean_validation_f1_macro": score, "folds": folds,
            })
            return score

        winner = search.run(study, self.config, evaluate, n_trials=self.config.search_trials)
        report = search.report(study)
        writer.write_json(f"search_test{test_signer}.json", report)
        trial_path = search_dir / f"trial_test{test_signer}_{winner.number}.json"
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
        candidate = search.configuration(self.config, winner.params)
        print(f"Vencedor: tentativa {winner.number + 1} | F1 validação={winner.value:.4f}", flush=True)
        return ModelSelection(
            candidate, trial["folds"], winner.value, str(trial_path.resolve()),
            validation_run_dir=self.runs_dir / trial["run_id"], search_report=report,
        )

    def _import_selection(self, rows, test_signer: str) -> ModelSelection:
        source = self.config.reuse_search.resolve()
        if source.is_file():
            source = source.parent
        metadata = json.loads((source / "search_config.json").read_text(encoding="utf-8"))
        base = metadata["base_config"]
        if Path(base["dataset"]).resolve() != self.dataset_root.resolve():
            raise ValueError("A busca importada pertence a outro dataset.")
        for name in ("model", "processor"):
            if base[name] != getattr(self.config, name):
                raise ValueError(f"A busca importada usa outro {name}.")
        divisions = metadata["divisions"]
        if test_signer not in divisions or set(divisions[test_signer]) != set(self._validation_signers(rows, test_signer)):
            raise ValueError("A busca importada usa outras divisões de teste/validação.")
        signature = metadata.get("dataset_signature")
        if signature is not None and signature != self._dataset_signature(rows):
            raise ValueError("Os dados mudaram desde a busca importada; execute uma nova busca.")

        report = json.loads((source / f"search_test{test_signer}.json").read_text(encoding="utf-8"))
        trial_path = source / f"trial_test{test_signer}_{report['best_trial']}.json"
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
        winning_config = trial["config"]
        old_processor = winning_config.get("processor_options", {})
        processor_defaults = {
            name: argument.default
            for name, argument in inspect.signature(ProcessorRegistry.get_class(self.config.processor)).parameters.items()
            if argument.default is not inspect.Parameter.empty
        }
        tuned_processor_options = set(metadata.get("parameters", {}).get("processor_options", {}))
        fixed_options = (old_processor.keys() | self.processor_options.keys()) - {"augmentation"} - tuned_processor_options
        for name in fixed_options:
            old_value = old_processor.get(name, processor_defaults.get(name))
            current_value = self.processor_options.get(name, processor_defaults.get(name))
            if old_value != current_value:
                raise ValueError(f"O processamento '{name}' difere da busca importada; execute uma nova busca.")

        folds = trial["folds"]
        if {fold["validation_signer"] for fold in folds} != set(divisions[test_signer]):
            raise ValueError("A tentativa vencedora não contém todas as validações da busca.")
        if any(fold["test_signer"] != test_signer for fold in folds):
            raise ValueError("A tentativa vencedora contém outro sinalizador de teste.")
        # Estudos antigos não registravam assinatura dos dados: conferimos os IDs do teste salvo.
        if signature is None:
            summary_path = source / "summary.json"
            if summary_path.exists():
                previous = json.loads(summary_path.read_text(encoding="utf-8"))
                expected = {row["sample_id"] for row in rows if row["signer_id"] == test_signer}
                for fold in previous["folds"]:
                    if fold["test_signer"] == test_signer and set(fold["test_samples"]) != expected:
                        raise ValueError("As amostras de teste mudaram desde a busca importada.")
            print("Busca antiga: assinatura completa dos dados indisponível; origem registrada no resultado.", flush=True)

        training_fields = HyperparameterSearch.PARAMETERS | {"model_options"}
        overrides = {name: winning_config[name] for name in training_fields if name in winning_config}
        processor_options = dict(self.config.processor_options)
        processor_options.update({name: old_processor[name] for name in tuned_processor_options if name in old_processor})
        overrides["processor_options"] = processor_options
        overrides.update({"search_config": None, "search_trials": None, "reuse_search": None,
                          "force_restart": False, "final_epochs": None})
        candidate = self.config.with_overrides(overrides)
        changes = {}
        if metadata.get("code_signature") != self._code_signature():
            changes["training_code"] = {"source": metadata.get("code_signature", "unverified"),
                                        "current": self._code_signature()}
        if winning_config.get("seed", base.get("seed", 42)) != candidate.seed:
            changes["seed"] = {"source": winning_config.get("seed", base.get("seed", 42)), "current": candidate.seed}
        if metadata.get("protocol_version", 1) < 2:
            changes["optimizer"] = {"source": "adam", "current": "adamw"}
        if old_processor.get("augmentation", False):
            changes["augmentation"] = {"source": True, "current": False}
        # Campos adicionados após a busca antiga usam os valores atuais, explicitamente registrados.
        for name in ("l1_lambda", "l2_lambda", "label_smoothing", "gradient_clip"):
            old_value = winning_config.get(name, 0.0 if name != "gradient_clip" else None)
            if old_value != getattr(candidate, name):
                changes[name] = {"source": old_value, "current": getattr(candidate, name)}
        if changes:
            print("Importando hiperparâmetros de outro protocolo; o F1 da origem não estima o treino atual.", flush=True)
        print(f"Reutilizando vencedor da busca: {trial_path}", flush=True)
        source_runs = source.parent
        return ModelSelection(
            candidate, folds, report["best_validation_f1_macro"], str(trial_path),
            validation_run_dir=source_runs / trial["run_id"], search_report=report, protocol_changes=changes,
        )

    def _fit_and_test(self, rows, test_signer: str, selection: ModelSelection, run_dir: Path) -> dict:
        config = selection.final_config(test_signer)
        pipeline = self._candidate_pipeline(config, rows)
        final_dir = run_dir / "final" / f"test_{test_signer}" / pipeline._run_id(rows)
        trainer = pipeline._writer(final_dir, rows)
        mapping = self._class_mapping(rows)
        train_rows = [row for row in rows if row["signer_id"] != test_signer]
        test_rows = [row for row in rows if row["signer_id"] == test_signer]
        trainer.config.update({"test_signer": test_signer, "train_signers": sorted({row['signer_id'] for row in train_rows})})
        trainer.write_json("config.json", config.to_dict())

        print(f"5. Treino final | {len(train_rows)} vídeos | {config.epochs} épocas", flush=True)
        training_path = final_dir / "training.json"
        if training_path.exists() and (final_dir / "final.pt").exists():
            training = json.loads(training_path.read_text(encoding="utf-8"))
            print("Reutilizando treino final concluído.", flush=True)
        else:
            pipeline._seed()
            if not (final_dir / "last.pt").exists():
                print("Inicializando um novo modelo com os pesos definidos na configuração.", flush=True)
            model = pipeline._create_model(rows, load_checkpoint=(final_dir / "last.pt").exists())
            training = trainer.fit(model, SignDataset(train_rows, pipeline.processor, mapping, training=True))

        print(f"6. Teste reservado | sinalizador {test_signer} | {len(test_rows)} vídeos", flush=True)
        test_path = final_dir / "test.json"
        if test_path.exists():
            test = json.loads(test_path.read_text(encoding="utf-8"))
            print("Reutilizando teste concluído.", flush=True)
        else:
            test = trainer.test(pipeline._create_model(rows, load_checkpoint=True), SignDataset(test_rows, pipeline.processor, mapping))

        result = {
            "test_signer": test_signer, "config": config.to_dict(), "epoch_selection": selection.epoch_selection(),
            "selection": {"source": selection.source, "inner_folds": selection.folds,
                          "score_scope": "source_validation", "protocol_changes": selection.protocol_changes,
                          "score_valid_for_current_protocol": selection.score is not None and not bool(selection.protocol_changes)},
            "validation_f1_macro": selection.score, "training": training,
            "test": test["metrics"], "test_samples": test["sample_ids"], "run_dir": str(final_dir.resolve()),
        }
        trainer.write_json("result.json", result)
        labels = {row["class_id"]: row.get("label", row["class_id"]) for row in rows}
        class_labels = [f"{class_id} · {labels[class_id]}" for class_id in mapping]
        histories = []
        if selection.validation_run_dir is not None:
            for fold in selection.folds:
                path = selection.validation_run_dir / f"test_{test_signer}" / f"val_{fold['validation_signer']}" / "history.json"
                if path.exists():
                    histories.append(json.loads(path.read_text(encoding="utf-8")))
        print(f"7. Resultados e gráficos: {final_dir}", flush=True)
        ExperimentReporter(final_dir, class_labels).write(
            json.loads((final_dir / "history.json").read_text(encoding="utf-8")), test,
            search_report=selection.search_report, validation_histories=histories,
        )
        return result

    def _summary(self, results: list[dict]) -> dict:
        summary = {"sessions": len(results), "folds": results, "metrics": {}}
        for name in ("f1_macro", "accuracy", "precision_macro", "recall_macro"):
            values = [result["test"][name] for result in results]
            summary["metrics"][name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        return summary
