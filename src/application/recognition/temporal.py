"""Janela temporal pequena, com relógio real e sem preencher mãos ausentes."""

from __future__ import annotations

from collections import deque
from math import ceil

import numpy as np

from application.types import Candidate


class LandmarkWindow:
    """Guarda somente os pontos necessários para uma janela de vídeo."""

    def __init__(self, seconds: float, sample_fps: float, max_gap: float) -> None:
        if min(seconds, sample_fps, max_gap) <= 0:
            raise ValueError("Duração, frequência e intervalo máximo devem ser positivos.")
        self.seconds = seconds
        self.sample_fps = sample_fps
        self.max_gap = max_gap
        # O limite de quantidade protege também contra uma fonte com FPS incorreto.
        self._samples: deque[tuple[float, np.ndarray]] = deque(
            maxlen=max(3, ceil(seconds * sample_fps * 4) + 2),
        )

    def clear(self) -> None:
        self._samples.clear()

    def append(self, timestamp: float, points: np.ndarray) -> bool:
        """Retorna True se uma interrupção obrigou a começar outra janela."""
        points = np.asarray(points, dtype=np.float32)
        if not np.isfinite(timestamp):
            raise ValueError("O instante do frame precisa ser finito.")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Os landmarks de um frame precisam ter formato (pontos, 2).")
        restarted = False
        if self._samples:
            previous_time, previous_points = self._samples[-1]
            if timestamp <= previous_time:
                raise ValueError("Os frames precisam chegar em ordem temporal crescente.")
            if points.shape != previous_points.shape:
                raise ValueError("A quantidade de landmarks mudou durante a captura.")
            if timestamp - previous_time > self.max_gap:
                self.clear()
                restarted = True
        sample = (timestamp, points.copy())
        # Uma fonte muito rápida não deve consumir o limite antes de completar
        # a duração. Conserva até dois pontos por intervalo de reamostragem.
        if len(self._samples) >= 2 and timestamp - self._samples[-2][0] < 1 / (2 * self.sample_fps):
            self._samples[-1] = sample
        else:
            self._samples.append(sample)
        start = timestamp - self.seconds
        # Preserva um ponto anterior à borda para reamostrar sem extrapolar.
        while len(self._samples) > 2 and self._samples[1][0] <= start:
            self._samples.popleft()
        return restarted

    @property
    def progress(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        duration = self._samples[-1][0] - self._samples[0][0]
        return min(1.0, duration / self.seconds)

    @property
    def ready(self) -> bool:
        return bool(self._samples) and self.progress >= 1.0 - 1e-9

    def resample(self) -> np.ndarray:
        """Grade uniforme agrupável em RGB, incluindo o instante mais recente.

        A frequência é nominal: 3 s × 30 FPS produzem 90 pontos no intervalo
        real. A quantidade é múltipla de três para o encoder não cortar o fim.
        Sequências continuam compatíveis e depois usam seu próprio frame_count.
        Um ponto ausente não vira uma trajetória fictícia entre observações.
        """
        if not self.ready:
            raise ValueError("A janela ainda não contém a duração necessária.")
        times = np.array([timestamp for timestamp, _ in self._samples])
        points = np.stack([points for _, points in self._samples])
        end = times[-1]
        count = max(3, round(self.seconds * self.sample_fps / 3) * 3)
        grid = np.linspace(end - self.seconds, end, count)
        right = np.searchsorted(times, grid, side="left").clip(0, len(times) - 1)
        left = (right - 1).clip(0)
        interval = times[right] - times[left]
        fraction = np.divide(
            grid - times[left], interval,
            out=np.zeros_like(grid), where=interval > 0,
        )[:, None, None]
        before, after = points[left], points[right]
        valid = np.isfinite(before).all(axis=2) & np.isfinite(after).all(axis=2)
        result = np.where(
            valid[:, :, None], before + (after - before) * fraction, np.nan,
        )
        exact = np.isclose(grid, times[right], rtol=0, atol=1e-9)
        result[exact] = after[exact]
        return result.astype(np.float32)


class PredictionStability:
    """Uma palavra só aparece após concordância entre previsões consecutivas."""

    def __init__(self, confidence: float, count: int) -> None:
        self.confidence = confidence
        self.required_count = count
        self.reset()

    def reset(self) -> None:
        self._class_id: str | None = None
        self._count = 0
        self.caption = ""

    def update(self, candidates: tuple[Candidate, ...]) -> str:
        if not candidates or candidates[0].confidence < self.confidence:
            self.reset()
            return ""
        best = candidates[0]
        self._count = self._count + 1 if best.class_id == self._class_id else 1
        self._class_id = best.class_id
        self.caption = best.label if self._count >= self.required_count else ""
        return self.caption
