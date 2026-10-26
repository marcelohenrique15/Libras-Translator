"""A janela pertence ao processo principal; o modo sem janela serve para verificação."""

import os
import sys
import time


class OpenCVDisplay:
    def __init__(self, title="Libras · Sinais ao vivo", *, width=1280, height=720):
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("O tamanho da janela deve conter inteiros positivos.")
        self.title = title
        self.width = width
        self.height = height
        self._opened = False

    def open(self):
        if sys.platform.startswith("linux") and not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")):
            raise RuntimeError("Sessão gráfica não encontrada. Use --headless para executar sem janela.")
        import cv2
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        self._opened = True
        cv2.resizeWindow(self.title, self.width, self.height)

    def show(self, image):
        import cv2
        cv2.imshow(self.title, image)

    def poll_key(self):
        import cv2
        key = cv2.waitKey(1)
        if self._opened:
            try:
                if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                    return 27
            except cv2.error:
                return 27
        return key & 0xFF if key >= 0 else -1

    def close(self):
        if self._opened:
            import cv2
            try:
                cv2.destroyWindow(self.title)
            except cv2.error:
                pass  # O usuário pode ter fechado a janela pelo gerenciador.
            self._opened = False


class HeadlessDisplay:
    def open(self):
        pass

    def show(self, image):
        pass

    def poll_key(self):
        time.sleep(0.005)
        return -1

    def close(self):
        pass
