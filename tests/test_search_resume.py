import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import optuna
from optuna.trial import TrialState

from cli.main import main
from tests import test_training_pipeline as fixtures
from training.search_resume import resume_latest_search
from training.trainer import Trainer
from training.training_pipeline import TrainingPipeline


class SearchResumeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingPipelineTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.config = self.fixture._search_config(trials=2)

    def _interrupt_search(self, fresh=False):
        pipeline = TrainingPipeline(self.config.with_overrides({"force_restart": fresh}))
        train_epoch = Trainer._train_epoch
        calls = 0

        def interrupt(trainer, model, optimizer, dataset):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt("Queda durante a segunda tentativa")
            return train_epoch(trainer, model, optimizer, dataset)

        with patch.object(Trainer, "_train_epoch", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                pipeline.train()
        return next(pipeline.runs_dir.glob("search_*/optuna.sqlite3"))

    def _study(self, database):
        return optuna.load_study(study_name="test_03", storage=f"sqlite:///{database.as_posix()}")

    def test_cli_resumes_latest_fresh_search_and_then_runs_final_training_and_test(self):
        database = self._interrupt_search(fresh=True)
        original = self._study(database)
        self.assertEqual([trial.state for trial in original.trials], [TrialState.COMPLETE, TrialState.RUNNING])
        first_parameters = original.trials[0].params.copy()
        pending_parameters = original.trials[1].params.copy()
        previous_best = original.best_value

        # Uma busca mais antiga não deve ser escolhida.
        older = database.parent.parent.parent / "search_older" / "optuna.sqlite3"
        older.parent.mkdir()
        older.touch()
        timestamp = database.stat().st_mtime_ns - 1_000_000_000
        os.utime(older, ns=(timestamp, timestamp))

        # O preset atual difere do original; a retomada deve restaurar o que foi salvo.
        preset = self.fixture.project_root / "preset.json"
        preset.write_text(json.dumps(self.config.with_overrides({"epochs": 99, "learning_rate": 0.1}).to_dict()))
        with patch.object(sys, "argv", [
            "libras-translator", "--train", "--resume-search", "--config", str(preset),
        ]):
            main()

        resumed = self._study(database)
        self.assertEqual(len(resumed.trials), 2)
        self.assertTrue(all(trial.state == TrialState.COMPLETE for trial in resumed.trials))
        self.assertEqual(resumed.trials[0].params, first_parameters)
        self.assertEqual(resumed.trials[1].params, pending_parameters)
        self.assertGreaterEqual(resumed.best_value, previous_best)
        summary = json.loads((database.parent / "summary.json").read_text())
        self.assertEqual(summary["folds"][0]["training"]["epochs_trained"], 1)
        self.assertEqual(summary["folds"][0]["config"]["epochs"], 1)
        self.assertTrue((Path(summary["folds"][0]["run_dir"]) / "test.json").exists())
        self.assertEqual(len(list(database.parent.parent.glob("search_*/optuna.sqlite3"))), 1)

    def test_resume_without_fresh_and_with_a_larger_total_trial_budget(self):
        database = self._interrupt_search()
        path = resume_latest_search(self.config, {"search_trials": 3})
        self.assertEqual(path.parent, database.parent)
        self.assertEqual(len(self._study(database).trials), 3)

    def test_missing_search_does_not_start_a_new_one(self):
        with patch.object(TrainingPipeline, "train", side_effect=AssertionError("Busca nova iniciada")):
            with self.assertRaisesRegex(ValueError, "Nenhuma busca"):
                resume_latest_search(self.config, {})

    def test_changed_training_settings_do_not_create_another_study(self):
        database = self._interrupt_search(fresh=True)
        metadata = (database.parent / "search_config.json").read_bytes()
        with patch.object(TrainingPipeline, "train", side_effect=AssertionError("Treino incompatível iniciado")):
            with self.assertRaisesRegex(ValueError, "configuração, código ou dados"):
                resume_latest_search(self.config, {"epochs": 2})
        self.assertEqual((database.parent / "search_config.json").read_bytes(), metadata)
        self.assertEqual(len(self._study(database).trials), 2)

    def test_changed_search_space_is_rejected_before_training(self):
        self._interrupt_search(fresh=True)
        self.config.search_config.write_text("n_trials = 2\nn_startup_trials = 1\n[parameters]\nlearning_rate = [0.1]\n")
        with patch.object(TrainingPipeline, "train", side_effect=AssertionError("Outra busca iniciada")):
            with self.assertRaisesRegex(ValueError, "espaço de busca"):
                resume_latest_search(self.config, {})

    def test_changed_dataset_is_rejected_before_training(self):
        self._interrupt_search(fresh=True)
        video = next((self.config.dataset / "raw").glob("*.mp4"))
        timestamp = video.stat().st_mtime_ns + 1_000_000
        os.utime(video, ns=(timestamp, timestamp))
        with patch.object(TrainingPipeline, "train", side_effect=AssertionError("Dados diferentes aceitos")):
            with self.assertRaisesRegex(ValueError, "configuração, código ou dados"):
                resume_latest_search(self.config, {})

    def test_cli_rejects_conflicting_actions_and_restart_flags(self):
        for arguments in (
            ["--encode-landmarks", "--resume-search"],
            ["--train", "--resume-search", "--force-restart"],
            ["--train", "--resume-search", "--reuse-search", "previous"],
        ):
            with self.subTest(arguments=arguments), patch.object(sys, "argv", ["libras-translator", *arguments]):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
