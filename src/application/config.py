"""Controles da aplicação; os parâmetros do modelo vêm do checkpoint."""

from dataclasses import dataclass, replace
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class RuntimeConfig:
    checkpoint: Path | None = None
    source: int | str = 0
    width: int = 960
    height: int = 540
    window_width: int = 1280
    window_height: int = 720
    fps: float = 30.0
    backend: int | None = None
    device: str = "auto"
    window_seconds: float = 3.0
    sample_fps: float = 30.0
    prediction_interval: float = 0.3
    confidence_threshold: float = 0.7
    stability_count: int = 3
    max_frame_gap: float = 0.35
    person_timeout: float = 0.6
    cpu_threads: int = 2
    max_prediction_age: float = 1.2
    mirror: bool = True

    def __post_init__(self):
        if self.checkpoint is not None:
            object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        if isinstance(self.source, bool) or not isinstance(self.source, (int, str)) or self.source == "":
            raise ValueError("source deve ser o índice inteiro da câmera ou o caminho de um vídeo.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device deve informar auto, cpu ou um dispositivo PyTorch.")
        for name in ("width", "height", "window_width", "window_height", "stability_count", "cpu_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} deve ser um inteiro positivo.")
        for name in ("fps", "window_seconds", "sample_fps", "prediction_interval", "max_frame_gap", "person_timeout", "max_prediction_age"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deve ser finito e positivo.")
        threshold = self.confidence_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("confidence_threshold deve estar entre 0 e 1.")
        if self.window_seconds * self.sample_fps < 3:
            raise ValueError("A janela deve conter pelo menos três frames.")
        if not isinstance(self.mirror, bool):
            raise ValueError("mirror deve ser true ou false.")

    @classmethod
    def from_toml(cls, path: Path):
        with Path(path).open("rb") as file:
            values = tomllib.load(file)
        unknown = values.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Opções desconhecidas na aplicação: {', '.join(sorted(unknown))}.")
        if values.get("checkpoint") == "auto":
            values["checkpoint"] = None
        return cls(**values)

    def with_overrides(self, **values):
        return replace(self, **{name: value for name, value in values.items() if value is not None})


def resolve_checkpoint(config: RuntimeConfig, runs_dir: Path = Path("runs")) -> RuntimeConfig:
    """Prefere o modelo final da execução concluída mais recente, sem olhar suas métricas."""
    if config.checkpoint is not None:
        path = config.checkpoint
    else:
        results = [path for path in runs_dir.rglob("result.json") if (path.parent / "final.pt").is_file()]
        if not results:
            raise FileNotFoundError("Nenhum modelo final encontrado em runs/. Informe --checkpoint caminho/final.pt.")
        latest = max(results, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        path = latest.parent / "final.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")
    return replace(config, checkpoint=path.resolve())
