import tempfile
import unittest
from pathlib import Path

from optuna.study import StudyDirection
from optuna.trial import TrialState

from training.config import TrainingConfig
from training.hyperparameter_search import HyperparameterSearch


class HyperparameterSearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)
        self.path = self.root / "search.toml"
        self.path.write_text(
            "n_trials = 3\nn_startup_trials = 2\n"
            "[parameters]\n"
            "learning_rate = { type = 'float', low = 0.0001, high = 0.01, log = true }\n"
            "batch_size = [2, 4]\n"
            "[parameters.model_options]\n"
            "hidden_size = [4, 8]\n"
            "[parameters.processor_options]\n"
            "frame_count = { type = 'int', low = 4, high = 8, step = 2 }\n"
        )
        self.search = HyperparameterSearch(self.path)
        self.config = TrainingConfig(
            dataset=self.root / "dataset", model="landmark_lstm",
            model_options={"hidden_size": 16, "dropout": 0.2},
            processor_options={"frame_count": 64, "augmentation": False},
            search_config=self.path, search_trials=3,
        )

    def _study(self):
        return self.search.create_study(self.root / "runs", "synthetic_f1", seed=42)

    def test_configuration_preserves_options_and_clears_search(self):
        candidate = self.search.configuration(self.config, {
            "learning_rate": 0.002, "batch_size": 4,
            "model_options.hidden_size": 8, "processor_options.frame_count": 6,
        })
        self.assertEqual(candidate.learning_rate, 0.002)
        self.assertEqual(candidate.batch_size, 4)
        self.assertEqual(candidate.model_options, {"hidden_size": 8, "dropout": 0.2})
        self.assertEqual(candidate.processor_options, {"frame_count": 6, "augmentation": False})
        self.assertIsNone(candidate.search_config)
        self.assertIsNone(candidate.search_trials)
        self.assertEqual(self.config.model_options["hidden_size"], 16)
        self.assertEqual(self.config.processor_options["frame_count"], 64)
        self.assertEqual(candidate.dataset, self.config.dataset)
        self.assertEqual(candidate.model, self.config.model)

    def test_study_maximizes_macro_f1_with_typed_parameters(self):
        study = self._study()
        scores = {}

        def evaluate(candidate, trial_number):
            self.assertIsInstance(candidate.learning_rate, float)
            self.assertGreaterEqual(candidate.learning_rate, 0.0001)
            self.assertLessEqual(candidate.learning_rate, 0.01)
            self.assertIn(candidate.batch_size, [2, 4])
            self.assertIn(candidate.model_options["hidden_size"], [4, 8])
            self.assertIsInstance(candidate.processor_options["frame_count"], int)
            self.assertIn(candidate.processor_options["frame_count"], [4, 6, 8])
            self.assertIsNone(candidate.search_config)
            self.assertIsNone(candidate.search_trials)
            score = 0.4 + candidate.batch_size * 0.05 + candidate.learning_rate
            scores[trial_number] = score
            return score

        winner = self.search.run(study, self.config, evaluate)
        self.assertEqual(study.direction, StudyDirection.MAXIMIZE)
        self.assertEqual(len(scores), 3)
        self.assertEqual(winner.value, max(scores.values()))
        self.assertEqual(winner.number, study.best_trial.number)
        self.assertEqual(winner.state, TrialState.COMPLETE)
        self.assertIn("model_options.hidden_size", winner.params)
        self.assertIn("processor_options.frame_count", winner.params)

    def test_completed_trials_are_reused_and_budget_is_total(self):
        study = self._study()
        evaluated = []

        def evaluate(candidate, trial_number):
            evaluated.append(trial_number)
            return 0.4 + trial_number * 0.05

        self.search.run(study, self.config, evaluate, n_trials=2)
        self.assertEqual(evaluated, [0, 1])
        loaded = self._study()
        self.assertEqual(len(loaded.trials), 2)
        winner = self.search.run(loaded, self.config, evaluate, n_trials=2)
        self.assertEqual(evaluated, [0, 1])
        self.assertEqual(winner.number, 1)
        self.search.run(loaded, self.config, evaluate, n_trials=5)
        self.assertEqual(evaluated, [0, 1, 2, 3, 4])
        self.assertEqual(len(loaded.get_trials(states=(TrialState.COMPLETE,))), 5)

    def test_interrupted_trial_resumes_same_configuration_before_new_trials(self):
        study = self._study()
        before = []

        def interrupt(candidate, trial_number):
            before.append((trial_number, candidate.to_dict()))
            if trial_number == 1:
                raise KeyboardInterrupt("interrupção simulada do treino")
            return 0.4

        with self.assertRaises(KeyboardInterrupt):
            self.search.run(study, self.config, interrupt, n_trials=3)
        self.assertEqual(study.trials[0].state, TrialState.COMPLETE)
        self.assertEqual(study.trials[1].state, TrialState.RUNNING)
        self.assertIn("config", study.trials[1].user_attrs)
        after = []

        def resume(candidate, trial_number):
            after.append((trial_number, candidate.to_dict()))
            return 0.5 + trial_number * 0.05

        loaded = self._study()
        winner = self.search.run(loaded, self.config, resume, n_trials=3)
        self.assertEqual([number for number, _ in after], [1, 2])
        self.assertEqual(after[0], before[1])
        self.assertEqual(len(loaded.trials), 3)
        self.assertTrue(all(trial.state == TrialState.COMPLETE for trial in loaded.trials))
        self.assertEqual(winner.value, max(trial.value for trial in loaded.trials))

    def test_search_space_cannot_change_dataset_model_or_test_split(self):
        for name, choices in (
            ("dataset", "['other_dataset']"),
            ("model", "['resnet18', 'landmark_lstm']"),
            ("test_signer_id", "['01', '02']"),
        ):
            with self.subTest(name=name):
                self.path.write_text(f"[parameters]\n{name} = {choices}\n")
                with self.assertRaises(ValueError):
                    HyperparameterSearch(self.path)


if __name__ == "__main__":
    unittest.main()
