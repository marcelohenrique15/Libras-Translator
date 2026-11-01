"""Coordena os componentes; não conhece landmarks ou camadas do modelo."""

from dataclasses import dataclass, field
import time

import numpy as np

from application.capture.camera import CameraSource
from application.config import RuntimeConfig
from application.ui.display import HeadlessDisplay, OpenCVDisplay
from application.ui.overlay import OverlayRenderer
from application.types import AnalysisResult, FramePacket
from application.runtime.worker import AnalysisWorker


class FrameRate:
    def __init__(self):
        self.value = 0.0
        self._previous = None

    def update(self, timestamp):
        if self._previous is not None and timestamp > self._previous:
            current = 1 / (timestamp - self._previous)
            self.value = current if not self.value else 0.8 * self.value + 0.2 * current
        self._previous = timestamp


@dataclass
class SessionStats:
    captured_frames: int = 0
    analyzed_frames: int = 0
    predictions: int = 0
    duration_seconds: float = 0.0
    last_candidates: list[dict] = field(default_factory=list)


class CameraApplication:
    """Interfaces pequenas permitem testar o controlador sem câmera nem GPU."""

    def __init__(
        self, config: RuntimeConfig, *, headless=False, max_seconds=None,
        camera=None, analysis=None, renderer=None, display=None,
    ):
        self.config = config
        self.max_seconds = max_seconds
        self.camera = camera if camera is not None else CameraSource(
            config.source, config.width, config.height, config.fps, config.backend,
        )
        self.analysis = analysis if analysis is not None else AnalysisWorker(config)
        self.renderer = renderer if renderer is not None else OverlayRenderer(
            config.mirror, config.max_prediction_age,
        )
        self.display = display if display is not None else (
            HeadlessDisplay() if headless else OpenCVDisplay(
                width=config.window_width, height=config.window_height,
            )
        )

    def run(self) -> SessionStats:
        started = time.monotonic()
        last_capture = started
        frame = FramePacket(-1, started, np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8))
        result = None
        paused = False
        capture_rate, analysis_rate = FrameRate(), FrameRate()
        stats = SessionStats()
        previous_prediction = None
        camera_started = False
        eof_since = None
        try:
            self.display.open()
            self.analysis.start()
            while True:
                cycle_start = time.monotonic()
                # Carregar o modelo antes da captura também preserva o início
                # de um arquivo de vídeo usado para reprodução e verificação.
                if self.analysis.ready and not camera_started:
                    self.camera.start()
                    camera_started = True
                    last_capture = cycle_start
                packet = self.camera.read_latest() if camera_started else None
                if packet is not None:
                    frame = packet
                    last_capture = cycle_start
                    capture_rate.update(packet.timestamp)
                    stats.captured_frames += 1
                    if not paused:
                        self.analysis.submit(packet)

                updated = self.analysis.read_latest()
                if updated is not None and not paused:
                    result = updated
                    stats.analyzed_frames += 1
                    analysis_rate.update(updated.timestamp)
                    if updated.prediction_timestamp is not None and updated.prediction_timestamp != previous_prediction:
                        previous_prediction = updated.prediction_timestamp
                        stats.predictions += 1
                        stats.last_candidates = [
                            {"label": item.label, "confidence": item.confidence} for item in updated.candidates
                        ]

                error = self.camera.error or self.analysis.error
                if error:
                    raise RuntimeError(error)
                if camera_started and cycle_start - last_capture > 5:
                    raise RuntimeError("A captura não respondeu por cinco segundos. Confira a câmera e o backend.")
                if not self.analysis.ready and cycle_start - started > 30:
                    raise RuntimeError("A análise não iniciou em 30 segundos. Confira o checkpoint e o dispositivo.")

                scene = result
                if not self.analysis.ready:
                    scene = AnalysisResult(frame.frame_id, cycle_start, None, status="Carregando modelo…")
                image = self.renderer.render(
                    frame, scene, device=self.analysis.device,
                    capture_fps=capture_rate.value, analysis_fps=analysis_rate.value, paused=paused,
                )
                self.display.show(image)
                key = self.display.poll_key()
                if key in (27, ord("q")):
                    break
                if key in (32, ord("r"), ord("R")):
                    if key == 32:
                        paused = not paused
                    self.analysis.reset()
                    self.renderer.reset()
                    result = None
                    previous_prediction = None
                if camera_started and self.camera.finished and packet is None:
                    eof_since = cycle_start if eof_since is None else eof_since
                    # Dá ao último resultado a chance de chegar; o prazo também
                    # cobre o descarte deliberado de um frame por fila cheia.
                    if result is not None and result.frame_id >= frame.frame_id:
                        break
                    if cycle_start - eof_since >= 1:
                        break
                if self.max_seconds is not None and time.monotonic() - started >= self.max_seconds:
                    break
                # A interface não gira sem limite quando a câmera/análise estão lentas.
                time.sleep(max(0, 1 / self.config.fps - (time.monotonic() - cycle_start)))
        finally:
            try:
                self.camera.stop()
            finally:
                try:
                    self.analysis.stop()
                finally:
                    self.display.close()
            stats.duration_seconds = time.monotonic() - started
        return stats
