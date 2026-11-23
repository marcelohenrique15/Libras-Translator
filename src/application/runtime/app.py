"""Coordena os componentes; não conhece landmarks ou camadas do modelo."""

from dataclasses import dataclass, field
import time

import numpy as np

from application.capture.camera import CameraSource
from application.capture.session import CaptureSession
from application.config import RuntimeConfig
from application.ui.display import HeadlessDisplay, OpenCVDisplay
from application.ui.overlay import OverlayRenderer
from application.ui.capture_overlay import CaptureRenderer
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
    first_prediction_seconds: float | None = None
    first_partial_caption_seconds: float | None = None
    first_stable_caption_seconds: float | None = None
    last_caption: str = ""
    capture_state: str = ""
    capture_status: str = ""
    capture_metrics: dict | None = None
    max_ui_interval_ms: float = 0.0


class CameraApplication:
    """Interfaces pequenas permitem testar o controlador sem câmera nem GPU."""

    def __init__(
        self, config: RuntimeConfig, *, headless=False, max_seconds=None, auto_capture=False,
        camera=None, analysis=None, renderer=None, display=None,
    ):
        self.config = config
        self.max_seconds = max_seconds
        self.auto_capture = auto_capture
        self.session = CaptureSession(config.capture_seconds) if config.mode == "manual" else None
        self.camera = camera if camera is not None else CameraSource(
            config.source, config.width, config.height, config.fps, config.backend,
        )
        self.analysis = analysis if analysis is not None else AnalysisWorker(config)
        renderer_class = CaptureRenderer if self.session is not None else OverlayRenderer
        self.renderer = renderer if renderer is not None else renderer_class(
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
        first_capture_time = None
        auto_capture_started = False
        previous_cycle = None
        try:
            self.display.open()
            self.analysis.start()
            while True:
                cycle_start = time.monotonic()
                if previous_cycle is not None:
                    stats.max_ui_interval_ms = max(stats.max_ui_interval_ms, (cycle_start - previous_cycle) * 1000)
                previous_cycle = cycle_start
                # Carregar o modelo antes da captura também preserva o início
                # de um arquivo de vídeo usado para reprodução e verificação.
                if self.analysis.ready and not camera_started:
                    self.camera.start()
                    camera_started = True
                    last_capture = cycle_start
                packet = self.camera.read_latest() if camera_started else None
                if self.session is not None:
                    self.session.scene(cycle_start)
                if packet is not None:
                    if first_capture_time is None:
                        first_capture_time = packet.timestamp
                    frame = packet
                    last_capture = cycle_start
                    capture_rate.update(packet.timestamp)
                    stats.captured_frames += 1
                    if self.auto_capture and not auto_capture_started and self.session is not None:
                        self._begin_capture(packet.timestamp)
                        auto_capture_started = True
                    if not paused and (self.session is None or self.session.state == "recording"):
                        self.analysis.submit(packet)
                if self.session is not None and self.session.state == "processing" and not self.session.finish_sent:
                    self.session.finish_sent = self.analysis.finish_capture(frame, self.session.ended_at)

                updated = self.analysis.read_latest()
                if updated is not None and not paused:
                    result = updated
                    stats.analyzed_frames += 1
                    analysis_rate.update(updated.timestamp)
                    if self.session is not None:
                        self.session.accept(updated)
                        if updated.capture_metrics is not None:
                            stats.capture_metrics = updated.capture_metrics
                    elapsed = max(0.0, time.monotonic() - first_capture_time) if first_capture_time is not None else 0.0
                    if updated.caption:
                        stats.last_caption = updated.caption
                        if updated.caption_is_partial and stats.first_partial_caption_seconds is None:
                            stats.first_partial_caption_seconds = elapsed
                        elif not updated.caption_is_partial and stats.first_stable_caption_seconds is None:
                            stats.first_stable_caption_seconds = elapsed
                    if updated.prediction_timestamp is not None and updated.prediction_timestamp != previous_prediction:
                        if stats.first_prediction_seconds is None:
                            stats.first_prediction_seconds = elapsed
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
                if self.session is not None:
                    scene = self.session.scene(cycle_start)
                    if self.session.state == "processing" and cycle_start - self.session.ended_at > 30:
                        raise RuntimeError("A classificação não respondeu em 30 segundos. Reinicie a aplicação.")
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
                if key in (ord("v"), ord("V")) and frame.frame_id >= 0 and camera_started and self.analysis.ready and self.session is not None:
                    if self.session.state == "recording":
                        self.session.stop(time.monotonic())
                    else:
                        self._begin_capture(time.monotonic())
                if key in (ord("r"), ord("R")) or (key == 32 and self.session is None):
                    if key == 32:
                        paused = not paused
                    self.analysis.reset()
                    self.renderer.reset()
                    result = None
                    previous_prediction = None
                    if self.session is not None:
                        self.session.cancel()
                if camera_started and self.camera.finished and packet is None:
                    eof_since = cycle_start if eof_since is None else eof_since
                    if self.session is not None:
                        if self.session.state == "recording":
                            self.session.reject(cycle_start, "Vídeo terminou antes de completar a captura.")
                            self.analysis.reset()
                        if not self.session.busy:
                            break
                    # Dá ao último resultado a chance de chegar; o prazo também
                    # cobre o descarte deliberado de um frame por fila cheia.
                    if self.session is None and result is not None and result.frame_id >= frame.frame_id:
                        break
                    if cycle_start - eof_since >= (10 if self.session is not None else 1):
                        if self.session is not None and self.session.busy:
                            self.session.reject(cycle_start, "Não houve resultado após o fim do vídeo.")
                            self.analysis.reset()
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
            if self.session is not None:
                stats.capture_state = self.session.state
                stats.capture_status = self.session.scene(time.monotonic()).status
        return stats

    def _begin_capture(self, timestamp: float) -> None:
        if self.session.start(timestamp):
            self.analysis.start_capture(timestamp)
            self.renderer.reset()
