"""Captura isolada: uma câmera lenta não bloqueia a janela da aplicação."""

from __future__ import annotations

import multiprocessing as mp
import math
import time
from queue import Empty
from typing import Any

from application.runtime.channels import put_latest
from application.types import FramePacket


def _replace_frame(queue: Any, packet: FramePacket) -> None:
    """Mantém no máximo um frame aguardando o consumidor, sem esperar por ele."""
    put_latest(queue, packet)


def _capture_worker(
    source: int | str,
    width: int,
    height: int,
    fps: float,
    backend: int | None,
    frames: Any,
    stop_event: Any,
    finished_event: Any,
    error_text: Any,
) -> None:
    """É o único proprietário do VideoCapture e de seus recursos nativos."""
    import cv2

    capture = None
    error = ""
    try:
        capture = (
            cv2.VideoCapture(source)
            if backend is None
            else cv2.VideoCapture(source, backend)
        )
        if not capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera ou vídeo: {source}.")

        is_file = isinstance(source, str)
        capture_fps = fps
        if is_file:
            recorded_fps = capture.get(cv2.CAP_PROP_FPS)
            if math.isfinite(recorded_fps) and recorded_fps > 0:
                capture_fps = recorded_fps
        else:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_id = 0
        consecutive_failures = 0
        next_capture = time.monotonic()
        while not stop_event.is_set():
            success, image = capture.read()
            captured_at = time.monotonic()
            if not success or image is None or image.size == 0:
                if is_file:
                    break
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    raise RuntimeError("A câmera parou de fornecer frames após 5 tentativas.")
                stop_event.wait(0.05)
                continue

            consecutive_failures = 0
            # Limita o custo do IPC mesmo quando a câmera ignora a resolução.
            # Mantém o enquadramento inteiro e nunca amplia uma imagem menor.
            image_height, image_width = image.shape[:2]
            scale = min(width / image_width, height / image_height)
            if scale < 1:
                size = (max(1, round(image_width * scale)), max(1, round(image_height * scale)))
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            _replace_frame(frames, FramePacket(frame_id, captured_at, image))
            frame_id += 1
            # VideoCapture de arquivo não espera o relógio. O mesmo limite evita
            # girar desnecessariamente quando um backend ignora o FPS pedido.
            next_capture = max(next_capture + 1 / capture_fps, captured_at)
            stop_event.wait(max(0.0, next_capture - time.monotonic()))
    except Exception as exception:
        error = str(exception) or type(exception).__name__
    finally:
        if capture is not None:
            try:
                capture.release()
            except Exception as exception:
                error = error or str(exception)
        if error:
            error_text.value = error[:1023]
        # EOF só é anunciado depois que o feeder terminou o último payload.
        # Se a interface já fechou, stop() do pai impõe o prazo de encerramento.
        frames.close()
        frames.join_thread()
        finished_event.set()


class CameraSource:
    """Entrega o frame mais recente sem bloquear a interface nem acumular vídeos."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 960,
        height: int = 540,
        fps: float = 30,
        backend: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Largura, altura e FPS devem ser positivos.")
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self._context = mp.get_context("spawn")
        self._process = None
        self._frames = None
        self._stop_event = None
        self._finished_event = None
        self._error_text = None
        self._error: str | None = None
        self._stopped = False

    def start(self) -> None:
        """Inicia uma vez; chamar novamente durante a captura não cria outro filho."""
        if self._process is not None:
            return
        self._error = None
        self._stopped = False
        self._frames = self._context.Queue(maxsize=1)
        self._stop_event = self._context.Event()
        self._finished_event = self._context.Event()
        # Um único escritor publica o diagnóstico antes de finished_event.
        # Sem lock: encerrar à força um filho não deixa um mutex preso no pai.
        self._error_text = self._context.RawArray("u", 1024)
        self._process = self._context.Process(
            target=_capture_worker,
            name="libras-camera",
            args=(
                self.source, self.width, self.height, self.fps, self.backend,
                self._frames, self._stop_event, self._finished_event, self._error_text,
            ),
            daemon=True,
        )
        try:
            self._process.start()
        except Exception:
            self._process = None
            self._frames.close()
            self._frames = None
            self._stopped = True
            raise

    def read_latest(self) -> FramePacket | None:
        """A fila tem capacidade 1: não há histórico para drenar indefinidamente."""
        if self._frames is None or self._stopped:
            return None
        try:
            packet = self._frames.get_nowait()
        except Empty:
            return None
        # A serialização já isolou esta imagem da memória do produtor.
        packet.image.setflags(write=False)
        return packet

    @property
    def error(self) -> str | None:
        if self._error_text is not None and self.finished:
            message = self._error_text.value
            if message:
                self._error = message
        if (
            self._error is None
            and not self._stopped
            and self._process is not None
            and self._process.exitcode not in (None, 0)
        ):
            self._error = f"A captura foi encerrada com código {self._process.exitcode}."
        return self._error

    @property
    def finished(self) -> bool:
        if self._stopped:
            return True
        if self._process is None:
            return False
        return self._finished_event.is_set() or self._process.exitcode is not None

    def stop(self) -> None:
        """Interrompe somente seu processo, mesmo se read() nativo não responder."""
        if self._stopped:
            return
        self._stopped = True
        if self._process is not None:
            self._stop_event.set()
            self._process.join(timeout=1.2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.4)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.4)
            # Copia o diagnóstico antes de liberar os canais.
            self.error
        if self._frames is not None:
            self._frames.cancel_join_thread()
            self._frames.close()
            self._frames = None

    def __enter__(self) -> CameraSource:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
