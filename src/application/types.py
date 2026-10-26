"""Mensagens entre captura, análise e interface, sem dependências de ML."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp: float
    image: np.ndarray


@dataclass(frozen=True)
class Candidate:
    class_id: str
    label: str
    confidence: float


@dataclass(frozen=True)
class PersonBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class AnalysisResult:
    frame_id: int
    timestamp: float
    box: PersonBox | None
    candidates: tuple[Candidate, ...] = ()
    caption: str = ""
    status: str = "Aguardando sinal"
    progress: float = 0.0
    inference_ms: float = 0.0
    analysis_ms: float = 0.0
    prediction_timestamp: float | None = None
