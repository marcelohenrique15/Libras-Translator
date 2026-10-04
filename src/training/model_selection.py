"""Configuração selecionada antes de acessar o conjunto de teste."""

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from training.config import TrainingConfig


@dataclass
class ModelSelection:
    config: TrainingConfig
    folds: list[dict]
    score: float | None
    source: str
    validation_run_dir: Path | None = None
    search_report: dict | None = None
    protocol_changes: dict = field(default_factory=dict)

    def epoch_selection(self) -> dict:
        if self.folds:
            best_epochs = [int(fold["best_epoch"]) for fold in self.folds]
            if any(epoch < 1 or epoch > self.config.epochs for epoch in best_epochs):
                raise ValueError("As melhores épocas devem estar dentro do orçamento da busca.")
            # Arredonda .5 para cima; a regra é definida antes do teste.
            epochs = int(median(best_epochs) + 0.5)
            return {"method": "median_best_epoch", "best_epochs": best_epochs, "final_epochs": epochs}
        epochs = self.config.final_epochs or self.config.epochs
        return {"method": "provided_final_epochs", "best_epochs": [], "final_epochs": epochs}

    def final_config(self, test_signer: str) -> TrainingConfig:
        epochs = self.epoch_selection()["final_epochs"]
        return self.config.with_overrides({
            "epochs": epochs, "final_epochs": epochs, "test_signer_id": test_signer,
            "search_config": None, "search_trials": None, "reuse_search": None, "force_restart": False,
        })
