"""Composição serial do detector, contexto incremental e política de legenda."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from application.config import RuntimeConfig
from application.recognition.captions import CaptionPolicy
from application.recognition.classifier import CheckpointClassifier
from application.recognition.landmarks import LandmarkDetector
from application.recognition.temporal import LandmarkWindow
from application.types import AnalysisResult, Candidate, FramePacket, PersonBox


class RecognitionPipeline:
    """Recebe frames em ordem, emite hipóteses e preserva a palavra estabilizada."""

    def __init__(self, config: RuntimeConfig, *, detector_factory: Callable | None = None,
                 classifier_factory: Callable | None = None) -> None:
        self.config = config
        self._detector_factory = detector_factory or LandmarkDetector
        self._classifier_factory = classifier_factory or CheckpointClassifier
        self._classifier = None
        self._detector = None
        self._window = LandmarkWindow(
            config.window_seconds, config.sample_fps, config.max_frame_gap,
            minimum_seconds=config.min_window_seconds,
        )
        self._captions = CaptionPolicy(
            confidence=config.confidence_threshold, count=config.stability_count,
            preview_confidence=config.preview_confidence_threshold,
            hold_seconds=config.caption_hold_seconds, partial_seconds=config.max_prediction_age,
        )
        self._hand_groups: list[list[int]] = []
        self._last_box: PersonBox | None = None
        self._caption_anchor: tuple[float, float] | None = None
        self._last_person_time: float | None = None
        self._last_frame_time: float | None = None
        self._anchor_missing_since: float | None = None
        self._last_hands_time: float | None = None
        self._segment_open = False
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
            # O custo da primeira chamada não deve interromper o primeiro gesto.
            warm_up = getattr(self._classifier, "warm_up", None)
            if callable(warm_up):
                warm_up()
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
            # Uma falha de tracking não deve apagar vários segundos de contexto.
            if self._last_person_time is not None:
                if frame.timestamp - self._last_person_time >= self.config.person_timeout:
                    self._clear_signal()
                    self._detector.reset()
                    self._last_person_time = None
                    self._last_box = None
                else:
                    self._window.append(frame.timestamp, observation.points)
            return self._result(frame, None, "Posicione-se diante da câmera", started)

        if self._last_box is not None and self._person_changed(self._last_box, box):
            self._clear_signal()
        self._last_box = box
        if observation.caption_anchor is not None:
            self._caption_anchor = observation.caption_anchor
        self._last_person_time = frame.timestamp
        if self._window.append(frame.timestamp, observation.points):
            self._clear_candidates()
            self._captions.reset()
        anchor_indices = self._classifier.processor.anchor_indices
        if anchor_indices and not np.isfinite(observation.points[anchor_indices]).all():
            if self._anchor_missing_since is None:
                self._anchor_missing_since = frame.timestamp
            if frame.timestamp - self._anchor_missing_since >= self.config.person_timeout:
                self._clear_signal()
            return self._result(frame, box, "Enquadre o rosto e o tronco", started)
        self._anchor_missing_since = None
        if self._prediction_time is not None and frame.timestamp - self._prediction_time > self.config.max_prediction_age:
            self._clear_candidates()

        hands_visible = observation.hands_present and self._visible_hand(observation.points)
        if not hands_visible:
            if self._segment_open and frame.timestamp - self._last_hands_time >= self.config.hand_pause_seconds:
                # Fecha uma única vez, aproveitando o gesto que acabou. Não exige
                # repetir a execução para conseguir outra observação do modelo.
                if self._window.ready:
                    self._infer(frame.timestamp, allow_confirmation=True)
                self._window.clear()
                self._segment_open = False
                self._last_hands_time = None
            return self._result(frame, box, "Pausa · aguardando próximo sinal", started)

        if not self._segment_open:
            self._window.keep_recent(min(0.3, self.config.min_window_seconds / 2))
            self._captions.break_agreement()
            self._segment_open = True
        self._last_hands_time = frame.timestamp
        if not self._window.ready:
            return self._result(frame, box, "Observando início do sinal", started)
        if self._prediction_time is None or frame.timestamp - self._prediction_time >= self.config.prediction_interval - 1e-9:
            self._infer(frame.timestamp, allow_confirmation=self._window.full)
        caption = self._captions.get(frame.timestamp)
        status = "Palavra estabilizada" if caption.text and not caption.is_partial else "Hipótese provisória · continue o sinal"
        return self._result(frame, box, status, started)

    def _infer(self, timestamp: float, *, allow_confirmation: bool) -> None:
        started = time.perf_counter()
        self._candidates = self._classifier.predict(self._window.resample())
        self._inference_ms = (time.perf_counter() - started) * 1000
        self._prediction_time = timestamp
        self._captions.update(self._candidates, timestamp, allow_confirmation=allow_confirmation)

    def _visible_hand(self, points: np.ndarray) -> bool:
        return any(np.isfinite(points[indices]).all(axis=1).mean() >= 0.6
                   for indices in self._hand_groups if indices)

    def _clear_candidates(self) -> None:
        self._candidates = ()
        self._prediction_time = None
        self._inference_ms = 0.0

    def _clear_signal(self) -> None:
        self._window.clear()
        self._clear_candidates()
        self._captions.reset()
        self._segment_open = False
        self._last_hands_time = None
        self._anchor_missing_since = None
        self._caption_anchor = None

    def _result(self, frame: FramePacket, box: PersonBox | None, status: str, started: float) -> AnalysisResult:
        caption = self._captions.get(frame.timestamp)
        # Oculta texto sem tracking, mas conserva o relógio para uma retomada
        # breve; perda prolongada e troca de pessoa já limpam a política.
        return AnalysisResult(
            frame_id=frame.frame_id, timestamp=frame.timestamp, box=box,
            candidates=self._candidates if box is not None else (),
            caption=caption.text if box is not None else "", status=status, progress=self._window.progress,
            inference_ms=self._inference_ms, analysis_ms=(time.perf_counter() - started) * 1000,
            prediction_timestamp=self._prediction_time,
            caption_is_partial=caption.is_partial, caption_timestamp=caption.timestamp,
            caption_expires_at=caption.expires_at,
            caption_anchor=self._caption_anchor if box is not None else None,
        )

    @staticmethod
    def _person_changed(before: PersonBox, after: PersonBox) -> bool:
        distance = np.hypot((before.left + before.right - after.left - after.right) / 2,
                            (before.top + before.bottom - after.top - after.bottom) / 2)
        separated = (before.right < after.left or after.right < before.left
                     or before.bottom < after.top or after.bottom < before.top)
        return bool(separated and distance > 0.35)
