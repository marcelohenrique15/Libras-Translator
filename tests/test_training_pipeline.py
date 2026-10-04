import csv
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from cli.main import main
from dataset.manifest_builder import ManifestBuilder
from dataset.manifest_splitter import ManifestSplitter
from dataset.sign_dataset import SignDataset
from models.model_registry import ModelRegistry
from models.sign_classifier import SignClassifier
from preprocessor.landmark_processor import LandmarkProcessor
from preprocessor.landmarks_extractor import LandmarksExtractor
from training.config import TrainingConfig
from training.trainer import Trainer
from training.training_pipeline import TrainingPipeline


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.project_root = Path(self.temporary_dir.name)
        self.root = self.project_root / "data" / "minds_libras"
        (self.root / "raw").mkdir(parents=True)
        (self.root / "processed" / "landmarks").mkdir(parents=True)
        for signer in ("01", "02", "03"):
            for class_id in ("01", "02"):
                sample_id = f"{class_id}SinalSinalizador{signer}-1"
                writer = cv2.VideoWriter(str(self.root / "raw" / f"{sample_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
                self.assertTrue(writer.isOpened())
                for _ in range(6):
                    writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
                writer.release()
                self._write_landmarks(self.root / "processed" / "landmarks" / f"{sample_id}.csv")
        ManifestSplitter().split(ManifestBuilder(self.root).build(), "03")
        self.config = TrainingConfig(
            dataset=self.root, epochs=1, batch_size=2, device="cpu",
            output_dir=self.project_root / "runs", weights_dir=self.project_root / "weights",
            model_options={"pretrained": False, "hidden_size": 8},
            processor_options={"augmentation": False},
        )
        self.pipeline = TrainingPipeline(self.config)
        self.rows = self.pipeline._prepare_manifest()
        self.mapping = {"01": 0, "02": 1}

    def _write_landmarks(self, path):
        columns = LandmarksExtractor(self.root)._columns()
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            for frame in range(6):
                coordinates = [0.2 + frame * 0.05, 0.4] * 543
                if frame == 2:
                    coordinates[0] = None
                writer.writerow([frame, *coordinates])

    def _external_config(self):
        """Um modelo mínimo mantém testes de orquestração rápidos e independentes."""
        module_name = "test_custom_experiment"
        source = self.project_root / f"{module_name}.py"
        source.write_text(
            "import numpy as np\nfrom torch import nn\n"
            "class Model(nn.Module):\n"
            "    INPUT_FORMAT = 'features'\n"
            "    def __init__(self, class_count):\n        super().__init__()\n        self.layer = nn.Linear(2, class_count)\n"
            "    def forward(self, inputs):\n        return self.layer(inputs)\n"
            "class Processor:\n"
            "    def __init__(self, dataset_root, augmentation=False, frame_count=1):\n"
            "        self.root = dataset_root\n        self.frame_count = frame_count\n"
            "    def prepare(self, sample_id):\n        return np.array([self.frame_count / 5, 0.5], dtype=np.float32)\n"
            "    def augment(self, data):\n        raise AssertionError('augmentation não deve ser chamada')\n"
            "    def encode(self, data):\n        return data\n"
            "    def process(self, sample_id):\n        return self.prepare(sample_id)\n"
            "    def prepare_all(self, rows, stage='encode'):\n        pass\n"
        )
        path_patch = patch.object(sys, "path", [str(self.project_root), *sys.path])
        path_patch.start()
        self.addCleanup(path_patch.stop)
        importlib.invalidate_caches()
        self.addCleanup(sys.modules.pop, module_name, None)
        return self.config.with_overrides({
            "model": f"{module_name}:Model", "processor": f"{module_name}:Processor",
            "model_options": {}, "processor_options": {},
        })

    def _search_config(self, trials=2):
        space = self.project_root / "search.toml"
        space.write_text(
            f"n_trials = {trials}\nn_startup_trials = 1\n[parameters]\n"
            "learning_rate = {type = 'float', low = 0.001, high = 0.002}\n"
        )
        return self._external_config().with_overrides({"search_config": space})

    def test_processor_stages_cache_and_input_formats(self):
        sample_id = self.rows[0]["sample_id"]
        processor = LandmarkProcessor(self.root)
        selected = processor.select(sample_id)
        prepared = processor.prepare(sample_id)
        self.assertTrue(np.isnan(selected).any())
        self.assertFalse(np.isnan(prepared).any())
        with patch.object(processor, "_read_landmarks", side_effect=AssertionError("seleção repetida")):
            np.testing.assert_array_equal(processor.prepare(sample_id), prepared)
        original = prepared.copy()
        train = SignDataset(self.rows, processor, self.mapping, training=True)
        validation = SignDataset(self.rows, processor, self.mapping)
        np.random.seed(42)
        self.assertEqual(train[0][0].shape, (3, 224, 224))
        self.assertTrue(torch.equal(train[0][0], validation[0][0]))
        np.testing.assert_array_equal(processor.prepare(sample_id), original)
        self.assertTrue(torch.equal(validation[0][0], validation[0][0]))
        for subset, count in (("asl_2nd", 80), ("all", 543), ("arcanjo", 75)):
            sequence = LandmarkProcessor(self.root, subset=subset, representation="sequence", frame_count=8)
            self.assertEqual(sequence.process(sample_id).shape, (8, count * 2))

        previous_id = self.pipeline._run_id(self.rows)
        source = processor.landmarks_dir / f"{sample_id}.csv"
        timestamp = max(source.stat().st_mtime_ns, (processor.cache_dir / f"{sample_id}.npy").stat().st_mtime_ns) + 1_000_000
        os.utime(source, ns=(timestamp, timestamp))
        with patch.object(processor, "_read_landmarks", wraps=processor._read_landmarks) as read:
            processor.prepare(sample_id)
            read.assert_called_once()
        self.assertNotEqual(previous_id, self.pipeline._run_id(self.rows))

    def test_cli_stages_reuse_inputs_and_can_train_without_raw_videos(self):
        with patch.object(LandmarksExtractor, "_extract_video", side_effect=AssertionError("CSV completo reextraído")):
            with patch.object(sys, "argv", ["libras-translator", "--select-landmarks", "--dataset", str(self.root)]):
                main()
            processor = LandmarkProcessor(self.root)
            sample_id = self.rows[0]["sample_id"]
            self.assertTrue((processor.selection_dir / f"{sample_id}.npy").exists())
            self.assertFalse(processor.imputation_dir.exists())
            with patch.object(sys, "argv", ["libras-translator", "--impute-landmarks", str(self.root)]):
                main()
            self.assertTrue((processor.imputation_dir / f"{sample_id}.npy").exists())
            self.assertFalse(processor.encoding_dir.exists())
            for video in (self.root / "raw").glob("*.mp4"):
                video.unlink()
            pipeline = TrainingPipeline(self.config)
            pipeline.prepare()
            self.assertEqual(pipeline.processor.process(sample_id).shape, (3, 224, 224))
            alias = TrainingPipeline(self.config.with_overrides({
                "processor": "preprocessor.landmark_processor:LandmarkProcessor",
            }))
            alias.prepare()
            self.assertEqual(alias.processor.representation, "image")
            self.assertEqual(alias.processor.process(sample_id).shape, (3, 224, 224))
            alias._run_id(self.rows)  # Usa os CSVs, mesmo sem vídeos originais.

    def test_partial_extraction_is_replaced_and_version_is_recorded(self):
        extractor = LandmarksExtractor(self.root)
        path = extractor.landmarks_dir / f"{self.rows[0]['sample_id']}.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:3]) + "\n")
        with patch.object(extractor, "_extract_video", side_effect=lambda video, output: self._write_landmarks(output)) as extract:
            extractor.extract()
            extract.assert_called_once()
        self.assertEqual(len(path.read_text().splitlines()), 7)
        index = json.loads(extractor.index_path.read_text())
        self.assertEqual(index["version"], extractor.version)
        self.assertEqual(len(index["samples"]), 6)

    def test_resnet_refit_is_fresh_and_uses_all_development_signers(self):
        models = []
        original_create = ModelRegistry.create
        original_train = Trainer.train
        original_fit = Trainer.fit
        refitted = []

        def create(name, class_count, parameters):
            model = original_create(name, class_count, parameters)
            models.append((model, {name: value.clone() for name, value in model.network.state_dict().items()}))
            return model

        def verify_split(trainer, model, train, validation):
            train_ids = {row["signer_id"] for row in train.rows}
            validation_ids = {row["signer_id"] for row in validation.rows}
            self.assertFalse(train_ids & validation_ids)
            self.assertNotIn("03", train_ids | validation_ids)
            self.assertTrue(train.training)
            self.assertFalse(validation.training)
            return original_train(trainer, model, train, validation)

        def verify_refit(trainer, model, dataset):
            self.assertEqual({row["signer_id"] for row in dataset.rows}, {"01", "02"})
            self.assertEqual(len(dataset), 4)
            initial = next(state for created, state in models if created is model)
            for name, current in model.network.state_dict().items():
                self.assertTrue(torch.equal(current, initial[name]), name)
            refitted.append(model)
            return original_fit(trainer, model, dataset)

        with patch.object(ModelRegistry, "create", side_effect=create):
            with patch.object(Trainer, "train", verify_split):
                with patch.object(Trainer, "fit", verify_refit):
                    path = self.pipeline.train()
        self.assertEqual(len(refitted), 1)
        # As camadas congeladas permanecem iguais também no treinamento final.
        for model, initial in models:
            if model not in refitted:
                continue
            for name, current in model.network.state_dict().items():
                if name.split(".")[0] not in model.trainable_layers:
                    self.assertTrue(torch.equal(current, initial[name]), name)
            self.assertFalse(torch.equal(model.network.layer4[0].conv1.weight, initial["layer4.0.conv1.weight"]))
        summary = json.loads(path.read_text())
        self.assertEqual(summary["sessions"], 1)
        self.assertEqual(len(summary["folds"]), 1)
        self.assertEqual(summary["folds"][0]["test_signer"], "03")
        final_dir = Path(summary["folds"][0]["run_dir"])
        self.assertTrue((final_dir / "final.pt").exists())
        self.assertEqual(path.parent.parent, self.project_root / "runs" / "minds_libras")
        self.assertFalse((self.root / "processed" / "training").exists())
        self.assertEqual(Path(torch.hub.get_dir()), self.config.weights_dir)
        with patch.object(ModelRegistry, "create", side_effect=AssertionError("modelo concluído recriado")):
            self.assertEqual(self.pipeline.train(), path)
        previous_id = self.pipeline._run_id(self.rows)
        with patch.dict(ModelRegistry.MODELS, {"another_model": SignClassifier}):
            self.assertEqual(self.pipeline._run_id(self.rows), previous_id)

    def test_external_model_and_processor_use_same_search_refit_protocol(self):
        result = TrainingPipeline(self._external_config()).train()
        summary = json.loads(result.read_text())
        self.assertEqual(summary["sessions"], 1)
        self.assertEqual(summary["folds"][0]["test_signer"], "03")
        self.assertEqual(len(summary["folds"][0]["test_samples"]), 2)
        self.assertTrue((Path(summary["folds"][0]["run_dir"]) / "final.pt").exists())

    def test_search_selects_by_validation_and_tests_only_the_winner(self):
        config = self._search_config()
        tested_rates = []
        searched = []
        refitted = []
        original_train = Trainer.train
        original_fit = Trainer.fit
        original_test = Trainer.test

        def verify_search(trainer, model, train, validation):
            train_ids = {row["signer_id"] for row in train.rows}
            validation_ids = {row["signer_id"] for row in validation.rows}
            self.assertFalse(train_ids & validation_ids)
            self.assertNotIn("03", train_ids | validation_ids)
            searched.append((train_ids, validation_ids))
            return original_train(trainer, model, train, validation)

        def verify_fit(trainer, model, train):
            self.assertEqual(len(searched), 4)
            self.assertEqual({row["signer_id"] for row in train.rows}, {"01", "02"})
            self.assertEqual(len(train), 4)
            refitted.append(trainer.config.copy())
            return original_fit(trainer, model, train)

        def verify_test(trainer, model, dataset):
            # O teste só é acessado depois da busca inteira e do refit único.
            self.assertEqual(len(searched), 4)
            self.assertEqual(len(refitted), 1)
            tested_rates.append(trainer.config["learning_rate"])
            self.assertEqual({row["signer_id"] for row in dataset.rows}, {"03"})
            return original_test(trainer, model, dataset)

        with patch.object(Trainer, "train", verify_search):
            with patch.object(Trainer, "fit", verify_fit):
                with patch.object(Trainer, "test", verify_test):
                    path = TrainingPipeline(config).train()
        report = json.loads((path.parent / "search_test03.json").read_text())
        winner = next(trial for trial in report["trials"] if trial["number"] == report["best_trial"])
        expected = max(trial["mean_validation_f1_macro"] for trial in report["trials"])
        self.assertEqual(winner["mean_validation_f1_macro"], expected)
        self.assertEqual(tested_rates, [winner["config"]["learning_rate"]])
        self.assertEqual(refitted[0]["learning_rate"], winner["config"]["learning_rate"])
        summary = json.loads(path.read_text())
        self.assertEqual(summary["sessions"], 1)
        self.assertEqual(summary["folds"][0]["epoch_selection"]["method"], "median_best_epoch")
        self.assertEqual(summary["folds"][0]["config"]["epochs"], 1)
        self.assertEqual(report["direction"], "maximize")
        self.assertEqual(report["objective"], "mean_validation_f1_macro")
        self.assertTrue((path.parent / "optuna.sqlite3").exists())
        self.assertEqual(len(list((self.project_root / "runs").rglob("test.json"))), 1)
        with patch.object(Trainer, "train", side_effect=AssertionError("candidato concluído repetido")):
            with patch.object(Trainer, "fit", side_effect=AssertionError("refit concluído repetido")):
                with patch.object(Trainer, "test", side_effect=AssertionError("teste repetido")):
                    self.assertEqual(TrainingPipeline(config).train(), path)
        expanded = TrainingPipeline(config.with_overrides({"search_trials": 3})).train()
        self.assertEqual(expanded, path)
        report = json.loads((path.parent / "search_test03.json").read_text())
        self.assertEqual(len(report["trials"]), 3)
        self.assertGreaterEqual(report["best_validation_f1_macro"], expected)

    def test_refit_epochs_come_from_validation_best_epochs(self):
        config = self._external_config().with_overrides({"epochs": 6})
        epochs = iter((3, 5))

        def validation_result(trainer, model, train, validation):
            return {
                "best_epoch": next(epochs), "validation_accuracy": 0.75,
                "validation_f1_macro": 0.7, "epochs_trained": 6,
            }

        original_fit = Trainer.fit
        fitted_epochs = []

        def verify_fit(trainer, model, dataset):
            fitted_epochs.append(trainer.config["epochs"])
            return original_fit(trainer, model, dataset)

        with patch.object(Trainer, "train", validation_result):
            with patch.object(Trainer, "fit", verify_fit):
                path = TrainingPipeline(config).train()
        final = json.loads(path.read_text())["folds"][0]
        self.assertEqual(fitted_epochs, [4])
        self.assertEqual(final["epoch_selection"]["best_epochs"], [3, 5])
        self.assertEqual(final["config"]["epochs"], 4)

    def test_all_test_signers_get_one_refit_and_one_test_each(self):
        config = self._external_config().with_overrides({"test_signer_id": "all"})
        original_test = Trainer.test
        tested = []

        def verify_test(trainer, model, dataset):
            signer = {row["signer_id"] for row in dataset.rows}
            self.assertEqual(len(signer), 1)
            tested.extend(signer)
            return original_test(trainer, model, dataset)

        with patch.object(Trainer, "test", verify_test):
            path = TrainingPipeline(config).train()
        summary = json.loads(path.read_text())
        self.assertEqual(summary["sessions"], 3)
        self.assertEqual(sorted(tested), ["01", "02", "03"])
        self.assertEqual(len(list(path.parent.rglob("final.pt"))), 3)

    def test_fixed_final_epochs_can_skip_validation_for_an_exported_configuration(self):
        config = self._external_config().with_overrides({"final_epochs": 2})
        original_fit = Trainer.fit
        fitted = []

        def record_fit(trainer, model, dataset):
            fitted.append(trainer.config["epochs"])
            return original_fit(trainer, model, dataset)

        with patch.object(Trainer, "train", side_effect=AssertionError("configuração final revalidada")):
            with patch.object(Trainer, "fit", record_fit):
                result = TrainingPipeline(config).train()
        self.assertEqual(fitted, [2])
        self.assertEqual(json.loads(result.read_text())["sessions"], 1)

    def test_force_restart_creates_fresh_run_and_preserves_previous_artifacts(self):
        config = self._search_config(trials=1)
        first_path = TrainingPipeline(config).train()
        first_summary = first_path.read_bytes()
        old_checkpoints = {path: path.read_bytes() for path in (self.project_root / "runs").rglob("*.pt")}
        original_train = Trainer.train
        trained_again = []

        def record_train(trainer, model, train, validation):
            trained_again.append(trainer.output_dir)
            return original_train(trainer, model, train, validation)

        with patch.object(Trainer, "train", record_train):
            second_path = TrainingPipeline(config.with_overrides({"force_restart": True})).train()
        self.assertNotEqual(second_path.parent, first_path.parent)
        self.assertEqual(len(trained_again), 2)
        self.assertEqual(first_path.read_bytes(), first_summary)
        for path, content in old_checkpoints.items():
            self.assertEqual(path.read_bytes(), content)

    def test_import_search_reuses_configuration_and_refits_without_search(self):
        config = self._search_config(trials=1)
        original_path = TrainingPipeline(config).train()
        imported = config.with_overrides({
            "reuse_search": original_path.parent, "search_config": None,
            "output_dir": self.project_root / "imported_runs",
        })
        original_fit = Trainer.fit
        refitted = []

        def verify_fit(trainer, model, train):
            self.assertEqual({row["signer_id"] for row in train.rows}, {"01", "02"})
            refitted.append(trainer.config.copy())
            return original_fit(trainer, model, train)

        with patch.object(Trainer, "train", side_effect=AssertionError("busca importada executada novamente")):
            with patch.object(Trainer, "fit", verify_fit):
                path = TrainingPipeline(imported).train()
        source = json.loads(original_path.read_text())["folds"][0]
        final = json.loads(path.read_text())["folds"][0]
        self.assertEqual(len(refitted), 1)
        self.assertEqual(final["config"]["learning_rate"], source["config"]["learning_rate"])
        self.assertEqual(final["config"]["epochs"], source["config"]["epochs"])
        self.assertTrue((Path(final["run_dir"]) / "final.pt").exists())

    def test_import_search_rejects_a_test_signer_not_in_source_search(self):
        config = self._search_config(trials=1)
        original_path = TrainingPipeline(config).train()
        imported = config.with_overrides({
            "reuse_search": original_path.parent, "search_config": None,
            "test_signer_id": "01", "output_dir": self.project_root / "imported_runs",
        })
        with self.assertRaises(ValueError):
            TrainingPipeline(imported).train()

    def test_import_search_rejects_changed_dataset_before_refit(self):
        config = self._search_config(trials=1)
        original_path = TrainingPipeline(config).train()
        video = self.root / self.rows[0]["path"]
        timestamp = video.stat().st_mtime_ns + 1_000_000
        os.utime(video, ns=(timestamp, timestamp))
        imported = config.with_overrides({
            "reuse_search": original_path.parent, "search_config": None,
            "output_dir": self.project_root / "imported_runs",
        })
        with patch.object(Trainer, "fit", side_effect=AssertionError("dataset incompatível treinado")):
            with self.assertRaisesRegex(ValueError, "dados mudaram"):
                TrainingPipeline(imported).train()

    def test_import_search_records_changed_code_and_seed_as_another_protocol(self):
        config = self._search_config(trials=1)
        original_path = TrainingPipeline(config).train()
        metadata_path = original_path.parent / "search_config.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["code_signature"] = "previous-training-code"
        metadata_path.write_text(json.dumps(metadata))
        imported = config.with_overrides({
            "reuse_search": original_path.parent, "search_config": None, "seed": config.seed + 1,
            "output_dir": self.project_root / "imported_runs",
        })
        pipeline = TrainingPipeline(imported)
        with patch.object(Trainer, "train", side_effect=AssertionError("busca importada repetida")):
            path = pipeline.train()
        result = json.loads(path.read_text())["folds"][0]
        changes = result["selection"]["protocol_changes"]
        self.assertEqual(changes["training_code"], {
            "source": "previous-training-code", "current": pipeline._code_signature(),
        })
        self.assertEqual(changes["seed"], {"source": config.seed, "current": imported.seed})
        self.assertFalse(result["selection"]["score_valid_for_current_protocol"])
        self.assertEqual(result["config"]["seed"], imported.seed)

    def test_import_search_applies_tuned_processor_option_to_refit_data(self):
        config = self._search_config(trials=1).with_overrides({"processor_options": {"frame_count": 4}})
        config.search_config.write_text(
            "n_trials = 1\nn_startup_trials = 1\n[parameters.processor_options]\nframe_count = [6]\n"
        )
        original_path = TrainingPipeline(config).train()
        imported = config.with_overrides({
            "reuse_search": original_path.parent, "search_config": None,
            "output_dir": self.project_root / "imported_runs",
        })
        original_fit = Trainer.fit
        processed_with = []

        def verify_fit(trainer, model, dataset):
            processed_with.append(dataset.processor.frame_count)
            np.testing.assert_array_equal(dataset[0][0].numpy(), np.array([1.2, 0.5], dtype=np.float32))
            return original_fit(trainer, model, dataset)

        with patch.object(Trainer, "train", side_effect=AssertionError("busca importada repetida")):
            with patch.object(Trainer, "fit", verify_fit):
                path = TrainingPipeline(imported).train()
        result = json.loads(path.read_text())["folds"][0]
        self.assertEqual(processed_with, [6])
        self.assertEqual(result["config"]["processor_options"]["frame_count"], 6)
        self.assertEqual(imported.processor_options["frame_count"], 4)
        self.assertTrue(result["selection"]["score_valid_for_current_protocol"])
        self.assertEqual(result["selection"]["protocol_changes"], {})

    def test_legacy_search_can_import_parameters_and_records_augmentation_change(self):
        config = self._search_config(trials=1)
        original_path = TrainingPipeline(config).train()
        metadata_path = original_path.parent / "search_config.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.pop("dataset_signature")
        metadata["base_config"]["processor_options"]["augmentation"] = True
        metadata_path.write_text(json.dumps(metadata))
        report = json.loads((original_path.parent / "search_test03.json").read_text())
        trial_path = original_path.parent / f"trial_test03_{report['best_trial']}.json"
        trial = json.loads(trial_path.read_text())
        trial["config"]["processor_options"]["augmentation"] = True
        for name in ("l1_lambda", "l2_lambda", "label_smoothing", "gradient_clip"):
            trial["config"].pop(name)
        trial_path.write_text(json.dumps(trial))

        imported = config.with_overrides({
            "reuse_search": original_path.parent, "search_config": None,
            "output_dir": self.project_root / "imported_runs",
        })
        with patch.object(Trainer, "train", side_effect=AssertionError("busca antiga repetida")):
            path = TrainingPipeline(imported).train()
        result = json.loads(path.read_text())["folds"][0]
        self.assertEqual(result["selection"]["protocol_changes"]["augmentation"], {"source": True, "current": False})
        self.assertFalse(result["selection"]["score_valid_for_current_protocol"])
        self.assertFalse(result["config"]["processor_options"].get("augmentation", False))
        self.assertEqual(result["validation_f1_macro"], report["best_validation_f1_macro"])

    def test_augmentation_is_disabled_and_cannot_be_enabled_in_experiment(self):
        processor = LandmarkProcessor(self.root)
        self.assertFalse(processor.augmentation)
        train = SignDataset(self.rows, processor, self.mapping, training=True)
        with patch.object(processor, "augment", create=True, side_effect=AssertionError("dados foram aumentados")):
            train[0]
        with self.assertRaises(ValueError):
            TrainingPipeline(self.config.with_overrides({"processor_options": {"augmentation": True}}))

    def test_cli_configuration_precedence(self):
        path = self.project_root / "experiment.toml"
        path.write_text("model = 'resnet18'\nepochs = 3\nbatch_size = 4\n[model_options]\nhidden_size = 8\n")
        args = [
            "libras-translator", "--train", "--dataset", str(self.root), "--config", str(path),
            "--epochs", "2", "--search-trials", "7", "--l1-lambda", "0.00001",
            "--l2-lambda", "0.0002", "--label-smoothing", "0.1", "--gradient-clip", "0.5",
            "--force-restart", "--reuse-search", str(self.project_root / "prior_search"),
        ]
        with patch.object(sys, "argv", args):
            with patch("training.training_pipeline.TrainingPipeline") as factory:
                main()
        selected = factory.call_args.args[0]
        self.assertEqual(selected.dataset, self.root)
        self.assertEqual(selected.model, "resnet18")
        self.assertEqual(selected.epochs, 2)
        self.assertEqual(selected.batch_size, 4)
        self.assertEqual(selected.model_options["hidden_size"], 8)
        self.assertEqual(selected.search_trials, 7)
        self.assertEqual(selected.l1_lambda, 0.00001)
        self.assertEqual(selected.l2_lambda, 0.0002)
        self.assertEqual(selected.label_smoothing, 0.1)
        self.assertEqual(selected.gradient_clip, 0.5)
        self.assertTrue(selected.force_restart)
        self.assertEqual(selected.reuse_search, self.project_root / "prior_search")


if __name__ == "__main__":
    unittest.main()
