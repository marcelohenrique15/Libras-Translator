"""Estado da captura acionada pelo usuário; não conhece câmera nem modelo."""

from dataclasses import replace

from application.types import AnalysisResult


class CaptureSession:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.cancel()

    @property
    def busy(self) -> bool:
        return self.state in {"recording", "processing"}

    def start(self, now: float) -> bool:
        if self.busy:
            return False
        self.started_at = now
        self.ended_at = None
        self.finish_sent = False
        self.state = "recording"
        self.result = None
        return True

    def cancel(self) -> None:
        self.state = "idle"
        self.started_at = None
        self.ended_at = None
        self.finish_sent = False
        self.result = None

    def stop(self, now: float) -> bool:
        """V novamente encerra o trecho; no processamento a tecla é ignorada."""
        if self.state != "recording":
            return False
        self.ended_at = min(max(now, self.started_at), self.started_at + self.seconds)
        self.state = "processing"
        return True

    def accept(self, result: AnalysisResult) -> None:
        if self.busy and result.capture_state in {"completed", "rejected"}:
            self.state = result.capture_state
            self.result = result

    def reject(self, now: float, reason: str) -> None:
        self.state = "rejected"
        self.result = AnalysisResult(-1, now, None, status=reason, capture_state="rejected")

    def scene(self, now: float) -> AnalysisResult:
        if self.result is not None:
            return replace(self.result, timestamp=now)
        progress = 0.0
        status = f"Pressione V para capturar {self.seconds:g} segundos"
        if self.busy:
            elapsed = max(0.0, now - self.started_at)
            progress = min(1.0, elapsed / self.seconds)
            if self.state == "recording" and elapsed >= self.seconds:
                self.stop(self.started_at + self.seconds)
            if self.state == "processing":
                status = "Processando o sinal…"
                progress = 1.0
            else:
                status = f"Faltam {self.seconds - elapsed:.1f} s · V para finalizar"
        return AnalysisResult(-1, now, None, status=status, progress=progress, capture_state=self.state)
