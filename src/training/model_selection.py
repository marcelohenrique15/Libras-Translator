"""Configuração selecionada antes de acessar o conjunto de teste."""

from dataclasses import dataclass, field
from pathlib import Path

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
    score_scope: str = "current_validation"

    def epoch_selection(self) -> dict:
        """Define os controles; a melhor época será escolhida na validação final."""
        return {
            "method": "final_validation_early_stopping", "monitor": "validation_f1_macro",
            "max_epochs": self.config.final_epochs or self.config.epochs,
            "patience": self.config.patience,
        }

    def final_config(self, test_signer: str) -> TrainingConfig:
        epochs = self.epoch_selection()["max_epochs"]
        return self.config.with_overrides({
            "epochs": epochs, "final_epochs": epochs, "test_signer_id": test_signer,
            "search_config": None, "search_trials": None, "reuse_search": None, "force_restart": False,
        })
