"""Desenha a interface sobre uma cópia da câmera, sem tocar na entrada do modelo."""

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from application.types import AnalysisResult, Candidate, FramePacket, PersonBox


class OverlayRenderer:
    """Três candidatos fixos no canto e uma legenda acompanhando a pessoa.

    Use uma instância somente na thread da interface. O espelhamento é visual;
    as coordenadas e a imagem recebidas continuam intactas para a inferência.
    """

    BACKGROUND = (30, 27, 23)  # OpenCV usa a ordem BGR.
    WHITE = (245, 246, 241)
    MUTED = (178, 182, 172)
    MINT = (181, 234, 120)
    CYAN = (238, 203, 110)

    def __init__(self, mirror: bool = True, max_result_age: float = 0.8) -> None:
        if max_result_age <= 0:
            raise ValueError("max_result_age deve ser positivo.")
        self.mirror = mirror
        self.max_result_age = max_result_age
        self._font_path = self._find_font()
        self._position: tuple[float, float] | None = None
        self._last_box: PersonBox | None = None
        self._prediction_cutoff = float("-inf")

    def reset(self) -> None:
        """Esquece a posição e o texto após limpar ou reiniciar a captura."""
        self._position = None
        self._last_box = None
        self._prediction_cutoff = float("-inf")

    def render(
        self,
        frame: FramePacket,
        result: AnalysisResult | None,
        *,
        device: str = "",
        capture_fps: float = 0.0,
        analysis_fps: float = 0.0,
        paused: bool = False,
        now: float | None = None,
    ) -> np.ndarray:
        """Retorna BGR uint8 pronto para imshow, sem modificar dados compartilhados."""
        image = cv2.flip(frame.image, 1) if self.mirror else frame.image.copy()
        height, width = image.shape[:2]
        now = monotonic() if now is None else now
        # Uma câmera inválida deve ser tratada pela captura. Imagens pequenas,
        # porém válidas, não comportam a interface e continuam sendo exibidas.
        if width < 160 or height < 120:
            return image

        scale = min(1.0, width / 850, height / 600)
        current = result is not None and 0 <= now - result.timestamp <= self.max_result_age
        status = "Pausado" if paused else (result.status if current else "Aguardando sinal")
        person = result.box if current else None
        show_prediction = False
        if person is None:
            self._position = None
            self._last_box = None
            # Uma predição antiga não reaparece ao detectar outra pessoa.
            self._prediction_cutoff = max(self._prediction_cutoff, now)
        else:
            prediction_time = result.prediction_timestamp
            if prediction_time is None:
                prediction_time = result.timestamp
            fresh_prediction = (
                prediction_time >= self._prediction_cutoff
                and 0 <= now - prediction_time <= self.max_result_age
            )
            # Um salto grande de posição também reinicia o acompanhamento.
            if self._last_box is not None and self._center_distance(self._last_box, person) > 0.4:
                self._position = None
                self._prediction_cutoff = result.timestamp
                fresh_prediction = fresh_prediction and prediction_time >= result.timestamp
            self._last_box = person
            show_prediction = fresh_prediction and not paused
            # Só a hipótese líder já confirmada acompanha a pessoa. Os demais
            # candidatos continuam no painel fixo, sem parecer uma legenda.
            if (show_prediction and result.candidates
                    and result.caption == result.candidates[0].label):
                self._draw_caption(image, person, result.caption, scale)

        candidates = result.candidates[:3] if show_prediction else ()
        progress = result.progress if current and not paused else 0.0
        self._draw_candidates(image, candidates, status, progress, scale)
        self._draw_footer(image, device, capture_fps, analysis_fps, scale, paused,
                          result=result if current else None, now=now)
        return image

    def _draw_caption(
        self, image: np.ndarray, person: PersonBox, caption: str, scale: float,
    ) -> None:
        height, width = image.shape[:2]
        font_size = max(12, round(24 * scale))
        padding = max(8, round(18 * scale))
        text_width = int(np.ceil(self._font(font_size).getlength(caption)))
        panel_width = min(width - 16, round(320 * scale), text_width + 2 * padding)
        panel_height = min(height - 16, max(32, round(52 * scale)))
        box = self._display_box(person)
        x, y = self._panel_position(box, panel_width, panel_height, width, height)
        self._rounded_panel(image, (x, y, x + panel_width, y + panel_height), round(14 * scale))
        self._text(image, caption, x + padding, y + max(8, round(14 * scale)), font_size,
                   self.WHITE, max_width=panel_width - 2 * padding)

    @staticmethod
    def _candidate_bounds(width: int, height: int, scale: float) -> tuple[int, int, int, int]:
        margin = max(8, round(18 * scale))
        panel_width = min(width - 2 * margin, round(286 * scale))
        panel_height = min(height - 2 * margin, round(177 * scale))
        return width - margin - panel_width, margin, width - margin, margin + panel_height

    def _draw_candidates(
        self, image: np.ndarray, candidates: tuple[Candidate, ...],
        status: str, progress: float, scale: float,
    ) -> None:
        """O painel permanece no mesmo canto, independentemente da pessoa."""
        height, width = image.shape[:2]
        x, y, end_x, end_y = self._candidate_bounds(width, height, scale)
        self._rounded_panel(image, (x, y, end_x, end_y), round(14 * scale))
        padding = max(6, round(16 * scale))
        panel_width = end_x - x
        left, right = x + padding, x + panel_width - padding
        self._text(image, "MAIORES CONFIANÇAS", left, y + round(14 * scale),
                   round(10 * scale), self.MUTED, max_width=right - left)
        for index in range(3):
            row_y = y + round((40 + index * 31) * scale)
            if index < len(candidates):
                candidate = candidates[index]
                confidence = float(np.clip(candidate.confidence, 0.0, 1.0))
                color = self.MINT if index == 0 else self.WHITE
                self._text(image, candidate.label, left, row_y, round(13 * scale), color,
                           max_width=right - left - round(55 * scale))
                self._text(image, f"{confidence:.0%}", right - round(43 * scale), row_y,
                           round(12 * scale), color)
                bar_y = row_y + round(19 * scale)
                cv2.line(image, (left, bar_y), (right, bar_y), (58, 60, 54), max(1, round(2 * scale)))
                cv2.line(image, (left, bar_y), (left + round((right - left) * confidence), bar_y),
                         self.MINT if index == 0 else self.CYAN, max(1, round(2 * scale)))
            else:
                self._text(image, "—", left, row_y, round(13 * scale), self.MUTED)

        self._text(image, status, left, y + round(145 * scale), round(10 * scale), self.MUTED,
                   max_width=right - left)
        # O progresso informa a coleta temporal, sem fingir uma probabilidade.
        progress = float(np.clip(progress, 0.0, 1.0))
        if not candidates and progress > 0:
            bottom = end_y - max(3, round(10 * scale))
            cv2.line(image, (left, bottom), (left + round((right - left) * progress), bottom),
                     self.CYAN, max(1, round(2 * scale)))

    def _draw_footer(
        self, image: np.ndarray, device: str,
        capture_fps: float, analysis_fps: float, scale: float, paused: bool,
        *, result: AnalysisResult | None, now: float,
    ) -> None:
        height, width = image.shape[:2]
        margin = max(8, round(18 * scale))
        footer_height = max(28, round(38 * scale))
        y = height - margin - footer_height
        self._rounded_panel(image, (margin, y, width - margin, y + footer_height), round(10 * scale))
        text_y = y + max(6, round(11 * scale))
        left = margin + round(13 * scale)
        right = width - margin - round(13 * scale)
        # Cada lado tem seu espaço; textos longos são abreviados antes da divisão.
        statistics_left = left + round(364 * scale) if width >= 850 else None
        shortcut_width = right - left
        if statistics_left is not None:
            shortcut_width = statistics_left - left - round(22 * scale)
        shortcut = "ESC sair  ·  ESPAÇO continuar  ·  R limpar" if paused else "ESC sair  ·  ESPAÇO pausar  ·  R limpar"
        self._text(image, shortcut, left, text_y, round(11 * scale), self.MUTED,
                   max_width=shortcut_width)
        if statistics_left is not None:
            statistics = f"{capture_fps:.0f} fps câmera  ·  {analysis_fps:.1f} análises/s"
            if device:
                statistics += f"  ·  {device.upper()}"
            if result is not None and result.box is not None and not paused:
                age_ms = max(0.0, now - result.timestamp) * 1000
                statistics += f"  ·  atraso {age_ms:.0f} ms"
                if result.inference_ms > 0:
                    statistics += f"  ·  modelo {result.inference_ms:.0f} ms"
            font_size = max(8, round(10 * scale))
            statistics = self._fit_text(statistics, font_size, right - statistics_left)
            statistics_width = int(np.ceil(self._font(font_size).getlength(statistics)))
            statistics_x = max(statistics_left, right - statistics_width)
            self._text(image, statistics, statistics_x, text_y,
                       font_size, self.MUTED, max_width=right - statistics_x)

    def _display_box(self, box: PersonBox) -> PersonBox:
        if self.mirror:
            return PersonBox(1 - box.right, box.top, 1 - box.left, box.bottom)
        return box

    def _panel_position(
        self, box: PersonBox, panel_width: int, panel_height: int, width: int, height: int,
    ) -> tuple[int, int]:
        """Prefere espaço livre ao lado da pessoa, preservando mãos e tronco."""
        scale = min(1.0, width / 850, height / 600)
        margin, gap = max(8, round(18 * scale)), max(8, round(12 * scale))
        x_min, x_max = margin, max(margin, width - margin - panel_width)
        y_min = min(max(0, height - panel_height), margin)
        y_max = max(y_min, height - margin - max(28, round(38 * scale)) - gap - panel_height)

        person_left, person_right = box.left * width, box.right * width
        person_top, person_bottom = box.top * height, box.bottom * height
        target_x = (person_left + person_right - panel_width) / 2
        target_y = (person_top + person_bottom - panel_height) / 2
        left_x, right_x = person_left - gap - panel_width, person_right + gap
        above_y, below_y = person_top - gap - panel_height, person_bottom + gap
        left_fits, right_fits = left_x >= x_min, right_x <= x_max

        if left_fits and (not right_fits or person_left > width - person_right):
            target_x, x_max = left_x, left_x
        elif right_fits:
            target_x, x_min = right_x, right_x
        elif above_y >= y_min:
            target_y, y_max = above_y, above_y
        elif below_y <= y_max:
            target_y, y_min = below_y, below_y
        else:
            target_y = person_bottom - panel_height - gap

        target = (float(np.clip(target_x, x_min, x_max)), float(np.clip(target_y, y_min, y_max)))
        if self._position is None:
            self._position = target
        else:
            self._position = tuple(previous + 0.3 * (current - previous)
                                   for previous, current in zip(self._position, target))
        # A suavização também respeita a área livre: não atravessa o corpo ao
        # trocar de lado ou quando o sinalizador se aproxima da placa.
        self._position = (float(np.clip(self._position[0], x_min, x_max)),
                          float(np.clip(self._position[1], y_min, y_max)))
        x = int(np.clip(round(self._position[0]), 0, max(0, width - panel_width)))
        y = int(np.clip(round(self._position[1]), 0, max(0, height - panel_height)))
        # A legenda móvel não invade o painel fixo do canto superior direito.
        fixed_left, _, fixed_right, fixed_bottom = self._candidate_bounds(width, height, scale)
        if x < fixed_right and x + panel_width > fixed_left and y < fixed_bottom + gap:
            y = min(round(y_max), fixed_bottom + gap)
            self._position = (self._position[0], float(y))
        return x, y

    @staticmethod
    def _center_distance(first: PersonBox, second: PersonBox) -> float:
        return float(np.hypot((first.left + first.right - second.left - second.right) / 2,
                              (first.top + first.bottom - second.top - second.bottom) / 2))

    def _text(
        self, image: np.ndarray, text: str, x: int, y: int,
        size: int, color: tuple[int, int, int], max_width: int | None = None,
    ) -> None:
        """Textos acentuados usam Pillow; os pequenos rótulos são cacheados."""
        size = max(8, size)
        text = str(text)[:512]
        if max_width is not None:
            text = self._fit_text(text, size, max_width)
        if not text:
            return
        label = self._label(text, size, color)
        height, width = image.shape[:2]
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + label.shape[1]), min(height, y + label.shape[0])
        if right <= left or bottom <= top:
            return
        label = label[top - y:bottom - y, left - x:right - x]
        alpha = label[:, :, 3:4].astype(np.float32) / 255
        region = image[top:bottom, left:right]
        region[:] = (label[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)

    @lru_cache(maxsize=256)
    def _fit_text(self, text: str, size: int, max_width: int) -> str:
        font = self._font(size)
        if font.getlength(text) <= max_width:
            return text
        if max_width <= font.getlength("…"):
            return ""
        while text and font.getlength(text + "…") > max_width:
            text = text[:-1]
        return text + "…"

    @lru_cache(maxsize=512)
    def _label(self, text: str, size: int, color: tuple[int, int, int]) -> np.ndarray:
        font = self._font(size)
        bounds = font.getbbox(text)
        label = Image.new("RGBA", (max(1, bounds[2] - bounds[0]), max(1, bounds[3] - bounds[1])), (0, 0, 0, 0))
        ImageDraw.Draw(label).text((-bounds[0], -bounds[1]), text, font=font, fill=(*color, 255))
        # Os valores são armazenados em BGR para compor diretamente com OpenCV.
        return np.asarray(label)

    @lru_cache(maxsize=16)
    def _font(self, size: int):
        return ImageFont.truetype(str(self._font_path), size) if self._font_path else ImageFont.load_default(size=size)

    @staticmethod
    def _find_font() -> Path | None:
        """Tenta caminhos conhecidos uma vez, sem varrer diretórios a cada frame."""
        candidates = [
            Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
        matplotlib = find_spec("matplotlib")
        if matplotlib and matplotlib.origin:
            candidates.append(Path(matplotlib.origin).parent / "mpl-data/fonts/ttf/DejaVuSans.ttf")
        return next((path for path in candidates if path.is_file()), None)

    def _rounded_panel(self, image: np.ndarray, bounds: tuple[int, int, int, int], radius: int) -> None:
        height, width = image.shape[:2]
        left, top, right, bottom = bounds
        left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
        if right <= left or bottom <= top:
            return
        region = image[top:bottom, left:right]
        mask = self._rounded_mask(right - left, bottom - top, radius)
        background = np.full_like(region, self.BACKGROUND)
        blended = cv2.addWeighted(region, 0.18, background, 0.82, 0)
        np.copyto(region, blended, where=mask[:, :, None] > 0)

    @staticmethod
    @lru_cache(maxsize=64)
    def _rounded_mask(width: int, height: int, radius: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        radius = min(max(0, radius), width // 2, height // 2)
        cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
        cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
        for x in (radius, width - radius - 1):
            for y in (radius, height - radius - 1):
                cv2.circle(mask, (x, y), radius, 255, -1, cv2.LINE_AA)
        return mask
