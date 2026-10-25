"""Retoma estudos existentes sem alterar a identificação dos experimentos."""

import json
from pathlib import Path

import optuna

from training.config import TrainingConfig
from training.hyperparameter_search import HyperparameterSearch
from training.training_pipeline import TrainingPipeline


def resume_latest_search(config: TrainingConfig, overrides: dict) -> Path:
    root = config.output_dir / config.dataset.resolve().name
    databases = [
        *root.glob("search_*/optuna.sqlite3"),
        *root.glob("fresh_*/search_*/optuna.sqlite3"),
    ]
    if not databases:
        raise ValueError(f"Nenhuma busca Optuna encontrada em {root}.")

    database = max(databases, key=lambda path: (path.stat().st_mtime_ns, str(path))).resolve()
    metadata_path = database.parent / "search_config.json"
    if not metadata_path.is_file():
        raise ValueError(f"A busca mais recente não tem search_config.json: {database.parent}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved = TrainingConfig.from_dict(metadata["base_config"])

    # A configuração salva é a referência; somente argumentos explícitos a substituem.
    updates = {name: value for name, value in overrides.items()
               if name not in {"output_dir", "force_restart", "reuse_search", "processor_options"}}
    processor_options = {**saved.processor_options, **overrides.get("processor_options", {})}
    updates.update({"processor_options": processor_options, "force_restart": False, "reuse_search": None})
    restored = saved.with_overrides(updates)
    pipeline = TrainingPipeline(restored)
    # Tanto execuções comuns quanto fresh_<id> mantêm seus diretórios originais.
    pipeline.runs_dir = database.parent.parent
    print(f"Busca mais recente: {database}\nConferindo configuração, código e dados...", flush=True)

    rows = pipeline._prepare_manifest()
    search = HyperparameterSearch(restored.search_config)
    divisions = {signer: pipeline._validation_signers(rows, signer) for signer in pipeline._test_signers(rows)}
    checks = {
        "configuração, código ou dados": pipeline._run_id(rows) == metadata.get("base_run"),
        "espaço de busca": search.parameters == metadata.get("parameters"),
        "n_startup_trials": search.n_startup_trials == metadata.get("n_startup_trials"),
        "divisões de validação/teste": divisions == metadata.get("divisions"),
        "versão do Optuna": optuna.__version__ == metadata.get("optuna_version"),
        "protocolo": metadata.get("protocol_version") == 2 and metadata.get("sampler") == "TPE",
    }
    changed = [name for name, matches in checks.items() if not matches]
    if changed:
        raise ValueError(
            f"A busca mais recente mudou em: {', '.join(changed)}. "
            "Restaure os valores originais para retomá-la; nenhum novo estudo foi iniciado."
        )

    storage = f"sqlite:///{database.as_posix()}"
    for study in optuna.get_all_study_summaries(storage=storage):
        if study.best_trial is not None:
            best = study.best_trial
            print(f"Melhor resultado salvo | {study.study_name} | tentativa {best.number + 1} | "
                  f"F1-macro validação={best.value:.4f}", flush=True)

    return pipeline.train()
