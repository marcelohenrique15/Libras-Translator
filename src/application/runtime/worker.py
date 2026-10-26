"""Um proprietário do modelo e do rastreador, isolado da interface."""

import multiprocessing as mp
from queue import Empty

from application.runtime.channels import put_latest, receive_latest
from application.config import RuntimeConfig
from application.types import AnalysisResult, FramePacket


def _analyze(config, inputs, outputs, stopped, ready, finished, error_text, device_text, factory):
    outputs.cancel_join_thread()
    try:
        # Importação tardia: o processo da câmera e a janela não carregam PyTorch.
        if factory is None:
            from application.recognition.pipeline import RecognitionPipeline
            factory = RecognitionPipeline
        with factory(config) as pipeline:
            device_text.value = pipeline.device[:31]
            ready.set()
            current_generation = None
            while not stopped.is_set():
                try:
                    generation, frame = inputs.get(timeout=0.1)
                except Empty:
                    continue
                if generation != current_generation:
                    pipeline.reset()
                    current_generation = generation
                frame.image.setflags(write=False)
                result = pipeline.process(frame)
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
        return put_latest(self._inputs, (self._generation, frame))

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
