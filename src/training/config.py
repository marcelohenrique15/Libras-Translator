from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import tomllib


@dataclass
class TrainingConfig:
    dataset: Path = Path("data/minds_libras")
    dataset_format: str = "auto"
    model: str = "resnet18"
    processor: str = "landmarks"
    test_signer_id: str | None = None
    validation_signer_id: str | None = None
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    patience: int = 5
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42
    output_dir: Path = Path("runs")
    weights_dir: Path = Path("weights")
    model_options: dict = field(default_factory=dict)
    processor_options: dict = field(default_factory=dict)
    search_config: Path | None = None
    search_trials: int | None = None

    def __post_init__(self) -> None:
        for name in ("dataset", "output_dir", "weights_dir", "search_config"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value))

        self.model_options = dict(self.model_options)
        self.processor_options = dict(self.processor_options)

        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size e patience devem ser >= 1.")
        if self.num_workers < 0:
            raise ValueError("num_workers deve ser >= 0.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate deve ser > 0 e weight_decay deve ser >= 0.")
        if self.search_trials is not None and self.search_trials < 1:
            raise ValueError("search_trials deve ser >= 1.")

    # Público
    @classmethod
    def from_toml(cls, path: Path) -> "TrainingConfig":
        with Path(path).open("rb") as file:
            values = tomllib.load(file)
        return cls.from_dict(values)

    @classmethod
    def from_dict(cls, values: dict) -> "TrainingConfig":
        return cls(**values)

    def to_dict(self) -> dict:
        values = asdict(self)
        for name in ("dataset", "output_dir", "weights_dir", "search_config"):
            if values[name] is not None:
                values[name] = str(values[name])
        return values

    def with_overrides(self, values: dict) -> "TrainingConfig":
        return replace(self, **values)
