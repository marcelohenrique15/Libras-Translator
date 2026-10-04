import json
from pathlib import Path
import tomllib

import optuna
from optuna.trial import TrialState

from training.config import TrainingConfig


class HyperparameterSearch:
    PARAMETERS = {
        "learning_rate", "weight_decay", "batch_size", "epochs", "patience",
        "l1_lambda", "l2_lambda", "label_smoothing", "gradient_clip",
    }

    def __init__(self, path: Path) -> None:
        with Path(path).open("rb") as file:
            options = tomllib.load(file)
        self.n_trials = options.get("n_trials", 20)
        self.n_startup_trials = options.get("n_startup_trials", 5)
        self.parameters = options["parameters"]
        if self.n_trials < 1 or self.n_startup_trials < 0:
            raise ValueError("n_trials deve ser positivo e n_startup_trials deve ser >= 0.")
        self._validate_parameters()

    # Público
    def create_study(self, output_dir: Path, name: str, seed: int) -> optuna.Study:
        output_dir.mkdir(parents=True, exist_ok=True)
        database = output_dir.resolve() / "optuna.sqlite3"
        study = optuna.create_study(
            study_name=name,
            storage=f"sqlite:///{database.as_posix()}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=self.n_startup_trials),
            pruner=optuna.pruners.NopPruner(),
            load_if_exists=True,
        )
        study.set_user_attr("objective", "mean_validation_f1_macro")
        return study

    def run(self, study: optuna.Study, base_config: TrainingConfig, evaluate, n_trials: int | None = None):
        target = n_trials if n_trials is not None else self.n_trials
        completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
        print(f"Busca Optuna | estudo: {study.study_name} | objetivo: maximizar F1-macro médio de validação", flush=True)
        print("Configuração da busca:\n" + json.dumps({
            "n_trials": target, "completed_trials": completed,
            "sampler": "TPE", "n_startup_trials": self.n_startup_trials,
            "parameters": self.parameters,
        }, indent=2, ensure_ascii=False), flush=True)
        while completed < target:
            number, config = self._next_trial(study, base_config)
            print(f"Optuna: tentativa {number + 1} | {completed}/{target} concluídas.", flush=True)
            print("Configuração desta tentativa:\n" + json.dumps(config.to_dict(), indent=2, ensure_ascii=False), flush=True)
            score = evaluate(config, number)
            study.tell(number, score)
            completed += 1
            print(
                f"F1-macro validação={score:.4f} | melhor F1-macro={study.best_value:.4f}",
                flush=True,
            )
        winner_config = self.configuration(base_config, study.best_trial.params)
        print(f"Melhor tentativa: {study.best_trial.number + 1}\nConfiguração vencedora da busca:\n"
              + json.dumps(winner_config.to_dict(), indent=2, ensure_ascii=False), flush=True)
        return study.best_trial

    def configuration(self, base_config: TrainingConfig, parameters: dict) -> TrainingConfig:
        overrides = {
            "search_config": None, "search_trials": None, "reuse_search": None,
            "force_restart": False, "final_epochs": None,
        }
        model_options = dict(base_config.model_options)
        processor_options = dict(base_config.processor_options)
        for name, value in parameters.items():
            if name.startswith("model_options."):
                # Optuna usa categorias escalares; o modelo recebe nomes de camadas em lista.
                if name == "model_options.trainable_layers" and isinstance(value, str):
                    value = [layer.strip() for layer in value.split(",") if layer.strip()]
                model_options[name.split(".", maxsplit=1)[1]] = value
            elif name.startswith("processor_options."):
                processor_options[name.split(".", maxsplit=1)[1]] = value
            else:
                overrides[name] = value
        overrides["model_options"] = model_options
        overrides["processor_options"] = processor_options
        return base_config.with_overrides(overrides)

    def report(self, study: optuna.Study) -> dict:
        return {
            "study": study.study_name,
            "direction": "maximize",
            "objective": "mean_validation_f1_macro",
            "best_trial": study.best_trial.number,
            "best_validation_f1_macro": study.best_value,
            "trials": [
                {
                    "number": trial.number, "state": trial.state.name,
                    "parameters": trial.params, "config": trial.user_attrs.get("config"),
                    "mean_validation_f1_macro": trial.value,
                }
                for trial in study.trials
            ],
        }

    # Privado
    def _next_trial(self, study: optuna.Study, base_config: TrainingConfig) -> tuple[int, TrainingConfig]:
        # Uma interrupção conserva os parâmetros e os checkpoints da tentativa.
        for trial in study.get_trials(states=(TrialState.RUNNING,)):
            if "config" in trial.user_attrs:
                print(f"Retomando tentativa {trial.number + 1}.", flush=True)
                return trial.number, self.configuration(base_config, trial.params)
            study.tell(trial.number, state=TrialState.FAIL)

        # Uma semente por tentativa permite retomar sem salvar o sampler em pickle.
        number = len(study.trials)
        study.sampler = optuna.samplers.TPESampler(
            seed=base_config.seed + number,
            n_startup_trials=self.n_startup_trials,
        )
        trial = study.ask()
        for name, specification in self.parameters.items():
            if name in {"model_options", "processor_options"}:
                for option, distribution in specification.items():
                    self._suggest(trial, f"{name}.{option}", distribution)
            else:
                self._suggest(trial, name, specification)
        config = self.configuration(base_config, trial.params)
        trial.set_user_attr("config", config.to_dict())
        return trial.number, config

    def _suggest(self, trial: optuna.Trial, name: str, distribution):
        if isinstance(distribution, list):
            return trial.suggest_categorical(name, distribution)
        options = dict(distribution)
        kind = options.pop("type")
        if kind == "float":
            return trial.suggest_float(name, **options)
        if kind == "int":
            return trial.suggest_int(name, **options)
        raise ValueError(f"Distribuição desconhecida para {name}: {kind}. Use float, int ou uma lista.")

    def _validate_parameters(self) -> None:
        if not self.parameters:
            raise ValueError("Defina pelo menos um hiperparâmetro em [parameters].")
        allowed = self.PARAMETERS | {"model_options", "processor_options"}
        unknown = set(self.parameters) - allowed
        if unknown:
            raise ValueError(f"Hiperparâmetros desconhecidos ou fixos na busca: {', '.join(sorted(unknown))}.")
