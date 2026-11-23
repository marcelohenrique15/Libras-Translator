"""Hipótese revisável e palavra estabilizada têm tempos de exibição distintos."""

from dataclasses import dataclass
from math import isfinite

from application.types import Candidate


@dataclass(frozen=True)
class CaptionState:
    text: str = ""
    is_partial: bool = False
    timestamp: float | None = None
    expires_at: float | None = None


class CaptionPolicy:
    """Atualiza texto a cada inferência, sem acumular palavras ou formar frases.

    Concordância estabiliza a apresentação; não prova que a classe está correta.
    O pipeline decide quando há contexto suficiente para permitir confirmação.
    """

    def __init__(
        self, confidence: float = 0.7, count: int = 2,
        preview_confidence: float = 0.5, hold_seconds: float = 2.0,
        partial_seconds: float = 0.6,
    ) -> None:
        if not 0 <= preview_confidence <= confidence <= 1:
            raise ValueError("Os limiares devem satisfazer 0 <= preview_confidence <= confidence <= 1.")
        if type(count) is not int or count < 1:
            raise ValueError("count deve ser um inteiro positivo.")
        if any(not isfinite(value) or value <= 0 for value in (hold_seconds, partial_seconds)):
            raise ValueError("Os tempos de exibição devem ser finitos e positivos.")
        self.confidence = confidence
        self.required_count = count
        self.preview_confidence = preview_confidence
        self.hold_seconds = hold_seconds
        self.partial_seconds = partial_seconds
        self.reset()

    def reset(self) -> None:
        self._state = CaptionState()
        self._caption_class: str | None = None
        self._agreement_class: str | None = None
        self._agreement_count = 0
        self._last_update: float | None = None

    def break_agreement(self) -> None:
        """Uma falha de detecção não apaga a palavra já estabilizada nem renova seu TTL."""
        self._agreement_class = None
        self._agreement_count = 0
        if self._state.is_partial:
            self._state = CaptionState()
            self._caption_class = None

    def update(
        self, candidates: tuple[Candidate, ...], timestamp: float,
        *, allow_confirmation: bool = True,
    ) -> CaptionState:
        """Recebe uma previsão nova; consultas de renderização usam somente get()."""
        if not isfinite(timestamp):
            raise ValueError("O instante da inferência deve ser finito.")
        if self._last_update is not None and timestamp <= self._last_update:
            raise ValueError("As inferências precisam chegar em ordem temporal crescente.")
        self._last_update = timestamp
        self.get(timestamp)
        if not candidates or not isfinite(candidates[0].confidence) or candidates[0].confidence < self.preview_confidence:
            self.break_agreement()
            return self._state

        best = candidates[0]
        if self._caption_class != best.class_id:
            # Uma hipótese nova substitui o texto anterior. Ele não fica guardado
            # para reaparecer quando a hipótese atual expirar.
            self._state = CaptionState()
            self._caption_class = best.class_id

        if best.confidence >= self.confidence:
            self._agreement_count = self._agreement_count + 1 if best.class_id == self._agreement_class else 1
            self._agreement_class = best.class_id
            if allow_confirmation and self._agreement_count >= self.required_count:
                self._state = CaptionState(best.label, False, timestamp, timestamp + self.hold_seconds)
                return self._state
        else:
            self._agreement_class = None
            self._agreement_count = 0

        # Uma confirmação ainda válida da mesma classe continua legível, mesmo
        # durante uma previsão mais fraca. Somente nova confirmação renova o TTL.
        if not self._state.text or self._state.is_partial:
            self._state = CaptionState(best.label, True, timestamp, timestamp + self.partial_seconds)
        return self._state

    def get(self, now: float) -> CaptionState:
        """Expira o texto sem renovar seus tempos ao redesenhar a interface."""
        if not isfinite(now):
            raise ValueError("O instante de consulta deve ser finito.")
        if self._state.expires_at is not None and now >= self._state.expires_at:
            self._state = CaptionState()
            self._caption_class = None
            self._agreement_class = None
            self._agreement_count = 0
        return self._state
