"""Um proprietário do modelo e do rastreador, isolado da interface."""

import multiprocessing as mp
from queue import Empty, Full

import numpy as np

from application.runtime.channels import put_latest, receive_latest
from application.config import RuntimeConfig
from application.types import AnalysisRequest, AnalysisResult, FramePacket


def _analyze(config, inputs, outputs, stopped, ready, finished, error_text, device_text, factory):
    outputs.cancel_join_thread()
    try:
        # Importação tardia: o processo da câmera e a janela não carregam PyTorch.
        if factory is None:
            if config.mode == "manual":
                from application.recognition.clip_pipeline import ClipRecognitionPipeline
                factory = ClipRecognitionPipeline
            else:
                from application.recognition.pipeline import RecognitionPipeline
                factory = RecognitionPipeline
        with factory(config) as pipeline:
            device_text.value = pipeline.device[:31]
            ready.set()
            current_generation = None
            while not stopped.is_set():
                try:
                    request = inputs.get(timeout=0.1)
                except Empty:
                    continue
                generation, frame = request.generation, request.frame
                if generation != current_generation:
                    if request.capture_started_at is not None:
                        pipeline.start_capture(request.capture_started_at)
                    else:
                        pipeline.reset()
                    current_generation = generation
                frame.image.setflags(write=False)
                result = pipeline.finish_capture(frame) if request.finish_capture else pipeline.process(frame)
                if result.capture_state in {"completed", "rejected"}:
                    # Um resultado final não pode ser perdido na corrida do
                    # feeder com uma atualização de progresso ainda pendente.
                    while not stopped.is_set():
                        try:
                            outputs.put((generation, result), timeout=.1)
                            break
                        except Full:
                            continue
                else:
                    put_latest(outputs, (generation, result))
    except Exception as error:
        error_text.value = (str(error) or type(error).__name__)[:2047]
    finally:
        finished.set()
        outputs.close()
        inputs.close()


class AnalysisWorker:
    """Só recebe snapshots; pausar/limpar invalida resultados ainda em trânsito."""

    def __init__(self, config: RuntimeConfig, pipeline_factory=None):
        self.config = config
        self._factory = pipeline_factory
        self._context = mp.get_context("spawn")
        self._process = None
        self._stopped = False
        self._generation = 0
        self._capture_started_at = None
        self._inputs = None
        self._outputs = None

    def start(self):
        if self._process is not None:
            return
        self._inputs = self._context.Queue(maxsize=1)
        self._outputs = self._context.Queue(maxsize=1)
        self._stop = self._context.Event()
        self._ready = self._context.Event()
        self._finished = self._context.Event()
        self._error_text = self._context.RawArray("u", 2048)
        self._device_text = self._context.RawArray("u", 32)
        self._process = self._context.Process(
            target=_analyze, name="libras-analysis",
            args=(self.config, self._inputs, self._outputs, self._stop, self._ready,
                  self._finished, self._error_text, self._device_text, self._factory),
            daemon=True,
        )
        try:
            self._process.start()
        except Exception:
            self._process = None
            self.stop()
            raise

    def submit(self, frame: FramePacket) -> bool:
        if self._inputs is None or self._stopped:
            return False
        return put_latest(self._inputs, AnalysisRequest(self._generation, frame, self._capture_started_at))

    def start_capture(self, timestamp: float):
        """A ordem acompanha todos os frames até ser recebida, sem fila extra."""
        self.reset()
        self._capture_started_at = timestamp

    def finish_capture(self, frame: FramePacket, ended_at: float | None = None) -> bool:
        if self._inputs is None or self._stopped:
            return False
        # Finalizar exige somente ID e instante. Não copia outra imagem nem
        # remove o último frame da fila; o controlador tenta depois se ela lotou.
        marker = FramePacket(frame.frame_id, frame.timestamp if ended_at is None else ended_at,
                             np.empty((0, 0, 3), dtype=np.uint8))
        try:
            self._inputs.put_nowait(AnalysisRequest(
                self._generation, marker, self._capture_started_at, finish_capture=True,
            ))
            return True
        except Full:
            return False

    def read_latest(self) -> AnalysisResult | None:
        if self._outputs is None or self._stopped:
            return None
        message = receive_latest(self._outputs)
        if message is None:
            return None
        generation, result = message
        return result if generation == self._generation else None

    def reset(self):
        # O número acompanha entrada e saída. Uma inferência iniciada antes de
        # pausar/limpar pode terminar, mas sua resposta nunca volta para a tela.
        self._generation += 1
        self._capture_started_at = None

    @property
    def ready(self):
        return self._process is not None and self._ready.is_set()

    @property
    def device(self):
        return self._device_text.value if self.ready else self.config.device

    @property
    def error(self):
        if self._process is None or self._stopped:
            return None
        if self._finished.is_set() and self._error_text.value:
            return self._error_text.value
        if self._process.exitcode not in (None, 0):
            return f"O processo de análise foi encerrado com código {self._process.exitcode}."
        return None

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        if self._process is not None:
            self._stop.set()
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.5)
        for channel in (self._inputs, self._outputs):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self._inputs = self._outputs = None
