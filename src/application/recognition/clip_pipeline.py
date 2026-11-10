"""Classifica uma captura completa, sem hipóteses nem segmentação automática."""

from dataclasses import replace
import time

import numpy as np

from application.recognition.classifier import CheckpointClassifier
from application.recognition.landmarks import LandmarkDetector
from application.recognition.temporal import LandmarkWindow
from application.types import AnalysisResult


class ClipRecognitionPipeline:
    def __init__(self, config, *, classifier_factory=None, detector_factory=None):
        self.config = config
        self._classifier_factory = classifier_factory or CheckpointClassifier
        self._detector_factory = detector_factory or LandmarkDetector
        self._classifier = None
        self._detector = None
        self.reset()

    @property
    def device(self):
        return self._classifier.device if self._classifier is not None else self.config.device

    def __enter__(self):
        self._classifier = self._classifier_factory(self.config)
        try:
            warm_up = getattr(self._classifier, "warm_up", None)
            if callable(warm_up):
                warm_up()
            self._detector = self._detector_factory(self._classifier.processor.landmark_groups)
        except Exception:
            self._classifier = None
            raise
        return self

    def __exit__(self, *_):
        try:
            if self._detector is not None:
                self._detector.close()
        finally:
            self._detector = self._classifier = None
            self.reset()

    def reset(self):
        self._started_at = None
        self._finished = None
        self._last_time = None
        self._sample_count = self._anchor_count = self._hand_count = 0
        self._gap = False
        self._max_gap = 0.0
        # O buffer não reinicia sozinho no meio da captura. Lacunas invalidam
        # o trecho explicitamente no fim, em vez de cortar o início do gesto.
        self._window = LandmarkWindow(self.config.capture_seconds, self.config.sample_fps,
                                      self.config.capture_seconds + 1,
                                      minimum_seconds=min(.6, self.config.capture_seconds * .8))
        # Reutiliza o rastreador já aquecido. Reiniciar seu grafo nativo a cada
        # gesto recria recursos; os pontos do trecho são limpos separadamente.

    def start_capture(self, timestamp: float):
        self.reset()
        self._started_at = timestamp

    def process(self, frame):
        if self._detector is None:
            raise RuntimeError("Abra ClipRecognitionPipeline com 'with' antes de capturar.")
        if self._finished is not None:
            return self._finished
        if self._started_at is None or frame.timestamp < self._started_at:
            return AnalysisResult(frame.frame_id, frame.timestamp, None, capture_state="idle")
        elapsed = frame.timestamp - self._started_at
        if elapsed >= self.config.capture_seconds:
            return self._finish(frame)
        if self._last_time is not None:
            if frame.timestamp <= self._last_time:
                raise ValueError("Frames de captura precisam chegar em ordem temporal crescente.")
            self._gap |= frame.timestamp - self._last_time > self.config.max_frame_gap
            self._max_gap = max(self._max_gap, frame.timestamp - self._last_time)
        else:
            self._gap |= frame.timestamp - self._started_at > self.config.max_frame_gap
        self._last_time = frame.timestamp
        observation = self._detector.process(frame.image)
        self._window.append(frame.timestamp, observation.points)
        self._sample_count += 1
        anchors = self._classifier.processor.anchor_indices
        self._anchor_count += int(observation.box is not None and (
            not anchors or np.isfinite(observation.points[anchors]).all()))
        self._hand_count += int(observation.hands_present)
        return AnalysisResult(frame.frame_id, frame.timestamp, observation.box,
                              progress=elapsed / self.config.capture_seconds, capture_state="recording")

    def _finish(self, frame):
        if self._finished is not None:
            return self._finished
        duration = max(0.0, min(frame.timestamp - self._started_at, self.config.capture_seconds)) if self._started_at is not None else 0.0
        metrics = {
            "duration_seconds": duration, "landmark_frames": self._sample_count,
            "observed_span_seconds": self._window.duration,
            "landmark_fps": (self._sample_count - 1) / self._window.duration if self._window.duration else 0.0,
            "hand_detection_fraction": self._hand_count / self._sample_count if self._sample_count else 0.0,
            "anchor_valid_fraction": self._anchor_count / self._sample_count if self._sample_count else 0.0,
            "max_landmark_gap_seconds": self._max_gap,
        }
        result = AnalysisResult(frame.frame_id, frame.timestamp, None, progress=1,
                                capture_state="rejected", capture_metrics=metrics)
        missing_end = self._last_time is None or self._started_at + duration - self._last_time > self.config.max_frame_gap
        if duration < min(.6, self.config.capture_seconds * .8) or self._sample_count < 6:
            result = replace(result, status="Captura muito curta. Faça o sinal completo antes de finalizar.")
        elif self._gap or missing_end or not self._window.ready:
            result = replace(result, status="Captura incompleta. Pressione V e tente novamente.")
        elif self._anchor_count < self._sample_count * .5 or self._hand_count < 3:
            result = replace(result, status="Enquadre rosto, tronco e mãos. Pressione V para repetir.")
        else:
            started = time.perf_counter()
            candidates = self._classifier.predict(self._window.resample())
            inference_ms = (time.perf_counter() - started) * 1000
            best = candidates[0]
            confident = best.confidence >= self.config.confidence_threshold
            result = replace(result, candidates=candidates,
                             caption=best.label if confident else f"Possível: {best.label}",
                             status="Concluído · V para outro sinal" if confident else "Baixa confiança · veja os candidatos",
                             prediction_timestamp=frame.timestamp, inference_ms=inference_ms,
                             caption_is_partial=not confident,
                             capture_state="completed")
        self._finished = result
        return result

    def finish_capture(self, frame):
        """Prazo encerrado pelo controlador, mesmo sem chegar outro frame."""
        return self._finish(frame)
