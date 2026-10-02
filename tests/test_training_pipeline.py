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

    def _lstm_config(self):
        return self.config.with_overrides({
            "model": "landmark_lstm", "model_options": {"hidden_size": 4},
            "processor_options": {"frame_count": 8},
        })

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
        self.assertFalse(torch.equal(train[0][0], validation[0][0]))
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
            pipeline = TrainingPipeline(self._lstm_config())
            pipeline.prepare()
            self.assertEqual(pipeline.processor.process(sample_id).shape, (8, 160))
            alias = TrainingPipeline(self._lstm_config().with_overrides({
                "processor": "preprocessor.landmark_processor:LandmarkProcessor",
            }))
            alias.prepare()
            self.assertEqual(alias.processor.representation, "sequence")
            self.assertEqual(alias.processor.process(sample_id).shape, (8, 160))
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

    def test_resnet_freezing_training_split_and_completed_fold_reuse(self):
        models = []
        original_create = ModelRegistry.create
        original_train = Trainer.train

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

        with patch.object(ModelRegistry, "create", side_effect=create):
            with patch.object(Trainer, "train", verify_split):
                path = self.pipeline.train()
        # Os dois primeiros modelos são treinados; os demais só avaliam checkpoints.
        for model, initial in models[:2]:
            for name, current in model.network.state_dict().items():
                if name.split(".")[0] not in model.trainable_layers:
                    self.assertTrue(torch.equal(current, initial[name]), name)
            self.assertFalse(torch.equal(model.network.layer4[0].conv1.weight, initial["layer4.0.conv1.weight"]))
        summary = json.loads(path.read_text())
        self.assertEqual(summary["sessions"], 2)
        self.assertEqual(path.parent.parent, self.project_root / "runs" / "minds_libras")
        self.assertFalse((self.root / "processed" / "training").exists())
        self.assertEqual(Path(torch.hub.get_dir()), self.config.weights_dir)
        with patch.object(ModelRegistry, "create", side_effect=AssertionError("modelo concluído recriado")):
            self.assertEqual(self.pipeline.train(), path)
        previous_id = self.pipeline._run_id(self.rows)
        with patch.dict(ModelRegistry.MODELS, {"another_model": SignClassifier}):
            self.assertEqual(self.pipeline._run_id(self.rows), previous_id)

    def test_lstm_and_new_external_model_use_the_same_trainer(self):
        pipeline = TrainingPipeline(self._lstm_config())
        path = pipeline.train()
        self.assertEqual(json.loads(path.read_text())["sessions"], 2)

        module_name = "test_custom_experiment"
        source = self.project_root / f"{module_name}.py"
        source.write_text(
            "import numpy as np\nfrom torch import nn\n"
            "class Model(nn.Module):\n"
            "    INPUT_FORMAT = 'features'\n"
            "    def __init__(self, class_count):\n        super().__init__()\n        self.layer = nn.Linear(2, class_count)\n"
            "    def forward(self, inputs):\n        return self.layer(inputs)\n"
            "class Processor:\n"
            "    def __init__(self, dataset_root):\n        self.root = dataset_root\n"
            "    def prepare(self, sample_id):\n        return np.array([0.2, 0.5], dtype=np.float32)\n"
            "    def augment(self, data):\n        return data\n"
            "    def encode(self, data):\n        return data\n"
            "    def process(self, sample_id):\n        return self.prepare(sample_id)\n"
            "    def prepare_all(self, rows, stage='encode'):\n        pass\n"
        )
        with patch.object(sys, "path", [str(self.project_root), *sys.path]):
            importlib.invalidate_caches()
            self.addCleanup(sys.modules.pop, module_name, None)
            config = self.config.with_overrides({
                "model": f"{module_name}:Model", "processor": f"{module_name}:Processor", "model_options": {},
            })
            result = TrainingPipeline(config).train()
            self.assertEqual(json.loads(result.read_text())["sessions"], 2)

    def test_search_selects_by_validation_and_tests_only_the_winner(self):
        space = self.project_root / "search.toml"
        space.write_text(
            "n_trials = 2\nn_startup_trials = 1\n[parameters]\n"
            "learning_rate = {type = 'float', low = 0.001, high = 0.002}\n"
        )
        config = self._lstm_config().with_overrides({"search_config": space})
        tested_rates = []
        original_test = Trainer.test

        def verify_test(trainer, model, dataset):
            # Todas as quatro rodadas de validação já terminaram antes de acessar teste.
            self.assertEqual(len(list((self.project_root / "runs").rglob("validation.json"))), 4)
            tested_rates.append(trainer.config["learning_rate"])
            self.assertEqual({row["signer_id"] for row in dataset.rows}, {"03"})
            return original_test(trainer, model, dataset)

        with patch.object(Trainer, "test", verify_test):
            path = TrainingPipeline(config).train()
        report = json.loads((path.parent / "search_test03.json").read_text())
        winner = next(trial for trial in report["trials"] if trial["number"] == report["best_trial"])
        expected = max(trial["mean_validation_f1_macro"] for trial in report["trials"])
        self.assertEqual(winner["mean_validation_f1_macro"], expected)
        self.assertEqual(tested_rates, [winner["config"]["learning_rate"]] * 2)
        self.assertEqual(report["direction"], "maximize")
        self.assertEqual(report["objective"], "mean_validation_f1_macro")
        self.assertTrue((path.parent / "optuna.sqlite3").exists())
        self.assertEqual(len(list((self.project_root / "runs").rglob("test.json"))), 2)
        with patch.object(Trainer, "train", side_effect=AssertionError("candidato concluído repetido")):
            with patch.object(Trainer, "test", side_effect=AssertionError("teste repetido")):
                self.assertEqual(TrainingPipeline(config).train(), path)
                one_fold = config.with_overrides({"validation_signer_id": "01"})
                other_path = TrainingPipeline(one_fold).train()
        self.assertNotEqual(other_path.parent, path.parent)
        self.assertEqual(json.loads(other_path.read_text())["sessions"], 1)
        self.assertEqual(json.loads(path.read_text())["sessions"], 2)
        expanded = TrainingPipeline(config.with_overrides({"search_trials": 3})).train()
        self.assertEqual(expanded, path)
        report = json.loads((path.parent / "search_test03.json").read_text())
        self.assertEqual(len(report["trials"]), 3)
        self.assertGreaterEqual(report["best_validation_f1_macro"], expected)

    def test_cli_configuration_precedence(self):
        path = self.project_root / "experiment.toml"
        path.write_text("model = 'landmark_lstm'\nepochs = 3\nbatch_size = 4\n[model_options]\nhidden_size = 8\n")
        args = ["libras-translator", "--train", "--dataset", str(self.root), "--config", str(path), "--epochs", "2", "--search-trials", "7"]
        with patch.object(sys, "argv", args):
            with patch("training.training_pipeline.TrainingPipeline") as factory:
                main()
        selected = factory.call_args.args[0]
        self.assertEqual(selected.dataset, self.root)
        self.assertEqual(selected.model, "landmark_lstm")
        self.assertEqual(selected.epochs, 2)
        self.assertEqual(selected.batch_size, 4)
        self.assertEqual(selected.model_options["hidden_size"], 8)
        self.assertEqual(selected.search_trials, 7)


if __name__ == "__main__":
    unittest.main()
