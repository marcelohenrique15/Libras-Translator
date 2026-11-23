"""Vídeo ao vivo, progresso de captura e legenda em uma faixa separada."""

import cv2
import numpy as np

from application.ui.overlay import OverlayRenderer


class CaptureRenderer(OverlayRenderer):
    def render(self, frame, result, *, device="", capture_fps=0.0, analysis_fps=0.0,
               paused=False, now=None):
        video = cv2.flip(frame.image, 1) if self.mirror else frame.image.copy()
        height, width = video.shape[:2]
        if width < 160 or height < 120:
            return video
        scale = min(1.0, width / 850, height / 600)
        subtitle_height = max(90, round(110 * scale))
        canvas = np.full((height + subtitle_height, width, 3), self.BACKGROUND, dtype=np.uint8)
        state = result.capture_state if result is not None else "idle"
        status = result.status if result is not None else "Pressione V para capturar"
        candidates = result.candidates if result is not None else ()
        progress = result.progress if result is not None else 0.0
        self._draw_candidates(video, candidates, status, progress, scale)
        margin = max(12, round(18 * scale))
        button_width = min(round(300 * scale), width - 2 * margin)
        self._rounded_panel(video, (margin, margin, margin + button_width, margin + round(47 * scale)), 12)
        button_text = {"recording": "[ V ]  FINALIZAR CAPTURA", "processing": "PROCESSANDO"}.get(state, "[ V ]  CAPTURAR SINAL")
        self._text(video, button_text, margin + 14, margin + 14, round(14 * scale),
                   self.MINT if state == "recording" else self.WHITE, max_width=button_width - 28)
        if state in {"recording", "processing"}:
            bar_y = height - margin
            cv2.line(video, (margin, bar_y), (width - margin, bar_y), (58, 60, 54), 4)
            cv2.line(video, (margin, bar_y), (margin + round((width - 2 * margin) * progress), bar_y), self.MINT, 4)
        canvas[:height] = video
        caption = result.caption if result is not None and result.caption else status
        size = max(16, round(29 * scale))
        caption = self._fit_text(caption, size, width - 2 * margin)
        text_width = self._font(size).getlength(caption)
        color = self.MINT if state == "completed" and not result.caption_is_partial else self.WHITE
        self._text(canvas, caption, round((width - text_width) / 2), height + 16, size, color)
        footer = f"V iniciar/finalizar  ·  R cancelar/limpar  ·  ESC sair  ·  {device.upper()}"
        if result is not None and result.inference_ms > 0:
            footer += f"  ·  classificação {result.inference_ms:.0f} ms"
        self._text(canvas, footer, margin, height + subtitle_height - 24, 11, self.MUTED,
                   max_width=width - 2 * margin)
        return canvas
