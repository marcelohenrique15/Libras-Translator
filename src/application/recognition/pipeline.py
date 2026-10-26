"""Composição serial do detector, janela temporal e classificador."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from application.config import RuntimeConfig
from application.recognition.classifier import CheckpointClassifier
from application.recognition.landmarks import LandmarkDetector
from application.recognition.temporal import LandmarkWindow, PredictionStability
from application.types import AnalysisResult, Candidate, FramePacket, PersonBox


class RecognitionPipeline:
    """Analisa frames em série; o controlador fornece sempre o mais recente."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        detector_factory: Callable | None = None,
        classifier_factory: Callable | None = None,
    ) -> None:
        self.config = config
        self._detector_factory = detector_factory or LandmarkDetector
        self._classifier_factory = classifier_factory or CheckpointClassifier
        self._classifier = None
        self._detector = None
        self._window = LandmarkWindow(
            config.window_seconds, config.sample_fps, config.max_frame_gap,
        )
        self._stability = PredictionStability(config.confidence_threshold, config.stability_count)
        self._hand_groups: list[list[int]] = []
        self._last_box: PersonBox | None = None
        self._last_person_time: float | None = None
        self._last_frame_time: float | None = None
        self._prediction_time: float | None = None
        self._candidates: tuple[Candidate, ...] = ()
        self._inference_ms = 0.0

    @property
    def device(self) -> str:
        return self._classifier.device if self._classifier is not None else self.config.device

    def __enter__(self) -> RecognitionPipeline:
        if self._classifier is not None:
            return self
        self._classifier = self._classifier_factory(self.config)
        try:
            processor = self._classifier.processor
            self._detector = self._detector_factory(processor.landmark_groups)
            offset = 0
            self._hand_groups = []
            for group, indices in processor.landmark_groups:
                if group in {"left_hand", "right_hand"}:
                    self._hand_groups.append(list(range(offset, offset + len(indices))))
                offset += len(indices)
        except Exception:
            if self._detector is not None:
                self._detector.close()
                self._detector = None
            self._classifier = None
            raise
        return self

    def __exit__(self, *_: object) -> None:
        try:
            if self._detector is not None:
                self._detector.close()
        finally:
            self._detector = None
            self._classifier = None
            self.reset()

    def reset(self) -> None:
        """Descarta o sinal anterior, inclusive após troca de pessoa ou de câmera."""
        self._clear_signal()
        self._last_box = None
        self._last_person_time = None
        self._last_frame_time = None
        if self._detector is not None:
            self._detector.reset()

    def process(self, frame: FramePacket) -> AnalysisResult:
        if self._detector is None or self._classifier is None:
            raise RuntimeError("Abra RecognitionPipeline com 'with' antes de analisar frames.")
        started = time.perf_counter()
        if self._last_frame_time is not None:
            if frame.timestamp <= self._last_frame_time:
                raise ValueError("Os frames precisam chegar em ordem temporal crescente.")
            if frame.timestamp - self._last_frame_time > self.config.max_frame_gap:
                self._clear_signal()
        self._last_frame_time = frame.timestamp
        observation = self._detector.process(frame.image)
        box = observation.box
        if box is None:
            self._clear_signal()
            if (
                self._last_person_time is not None
                and frame.timestamp - self._last_person_time >= self.config.person_timeout
            ):
                self._detector.reset()
                self._last_person_time = None
                self._last_box = None
            return self._result(frame, None, "Posicione-se diante da câmera", started)

        if self._last_box is not None and self._person_changed(self._last_box, box):
            self._clear_signal()
        self._last_box = box
        self._last_person_time = frame.timestamp
        anchor_indices = self._classifier.processor.anchor_indices
        if anchor_indices and not np.isfinite(observation.points[anchor_indices]).all():
            self._clear_signal()
            return self._result(frame, box, "Enquadre o rosto e o tronco", started)

        if self._window.append(frame.timestamp, observation.points):
            self._clear_prediction()
        if not observation.hands_present or not self._visible_hand(observation.points):
            # O treino inclui frames anteriores/posteriores ao sinal. Preserva
            # rosto/corpo e mãos NaN na janela, mas nunca uma legenda antiga.
            self._clear_prediction()
            return self._result(frame, box, "Mostre as mãos para começar", started)
        if (
            self._prediction_time is not None
            and frame.timestamp - self._prediction_time > self.config.max_prediction_age
        ):
            self._clear_prediction()
        if not self._window.ready:
            return self._result(frame, box, "Observando o sinal", started)
        if (
            self._prediction_time is None
            or frame.timestamp - self._prediction_time >= self.config.prediction_interval - 1e-9
        ):
            inference_started = time.perf_counter()
            self._candidates = self._classifier.predict(self._window.resample())
            self._inference_ms = (time.perf_counter() - inference_started) * 1000
            self._prediction_time = frame.timestamp
            self._stability.update(self._candidates)
        status = "Sinal reconhecido" if self._stability.caption else "Aguardando confirmação do sinal"
        return self._result(frame, box, status, started)

    def _visible_hand(self, points: np.ndarray) -> bool:
        return any(
            np.isfinite(points[indices]).all(axis=1).mean() >= 0.6
            for indices in self._hand_groups if indices
        )

    def _clear_prediction(self) -> None:
        self._candidates = ()
        self._prediction_time = None
        self._inference_ms = 0.0
        self._stability.reset()

    def _clear_signal(self) -> None:
        self._window.clear()
        self._clear_prediction()

    def _result(
        self, frame: FramePacket, box: PersonBox | None, status: str, started: float,
    ) -> AnalysisResult:
        return AnalysisResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            box=box,
            candidates=self._candidates,
            caption=self._stability.caption,
            status=status,
            progress=self._window.progress,
            inference_ms=self._inference_ms,
            analysis_ms=(time.perf_counter() - started) * 1000,
            prediction_timestamp=self._prediction_time,
        )

    @staticmethod
    def _person_changed(before: PersonBox, after: PersonBox) -> bool:
        # Uma mudança brusca reinicia a janela; não é identificação de pessoas.
        distance = np.hypot(
            (before.left + before.right - after.left - after.right) / 2,
            (before.top + before.bottom - after.top - after.bottom) / 2,
        )
        separated = (
            before.right < after.left or after.right < before.left
            or before.bottom < after.top or after.bottom < before.top
        )
        return bool(separated and distance > 0.35)
