"""Latência de classificação com lote unitário e entradas do dataset de teste."""

from datetime import datetime, timezone
import platform
from time import perf_counter

import numpy as np
import torch


class InferenceBenchmark:
    """Separa leitura/preparo, transferência, modelo e retorno da saída à CPU.

    Não usa DataLoader com prefetch: cada tempo pertence à mesma amostra.
    CUDA é sincronizada nas fronteiras para medir trabalho concluído.
    """

    def __init__(self, device: torch.device, warmup_iterations: int = 3):
        self.device = torch.device(device)
        self.warmup_iterations = warmup_iterations

    def run(self, model, dataset, *, after_accuracy_evaluation: bool = False) -> dict:
        if not len(dataset):
            raise ValueError("O benchmark precisa de pelo menos uma amostra.")
        model.to(self.device).eval()
        rows = []
        with torch.inference_mode():
            inputs = dataset[0][0].unsqueeze(0).to(self.device)
            for _ in range(self.warmup_iterations):
                self._predict(model, inputs).cpu()
            self._synchronize()
            started = perf_counter()
            for index in range(len(dataset)):
                sample_started = perf_counter()
                inputs = dataset[index][0].unsqueeze(0)
                loaded = perf_counter()
                inputs = inputs.to(self.device)
                self._synchronize()
                transferred = perf_counter()
                output = self._predict(model, inputs)
                self._synchronize()
                inferred = perf_counter()
                output.cpu().tolist()
                self._synchronize()
                finished = perf_counter()
                rows.append({
                    "sample_id": dataset.rows[index]["sample_id"],
                    "input_shape": list(inputs.shape), "input_dtype": str(inputs.dtype),
                    "data_ms": (loaded - sample_started) * 1000,
                    "transfer_ms": (transferred - loaded) * 1000,
                    "inference_ms": (inferred - transferred) * 1000,
                    "postprocess_ms": (finished - inferred) * 1000,
                    "processing_ms": (finished - sample_started) * 1000,
                })
            elapsed = perf_counter() - started
        stages = {
            name: self._statistics([row[name] for row in rows])
            for name in ("data_ms", "transfer_ms", "inference_ms", "postprocess_ms", "processing_ms")
        }
        return {
            "schema_version": 1,
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else (platform.processor() or platform.machine()),
            "platform": platform.platform(), "python_version": platform.python_version(),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "batch_size": 1, "num_workers": 0, "warmup_iterations": self.warmup_iterations,
            "after_accuracy_evaluation": after_accuracy_evaluation,
            "sample_count": len(rows), "duration_seconds": elapsed,
            "throughput_samples_per_second": len(rows) / elapsed,
            "inference_samples_per_second": 1000 / stages["inference_ms"]["mean"],
            "timings": stages, "samples": rows,
            "scope": "Separate benchmark pass with model warmup; OS/dataset caches are not "
                     "flushed or guaranteed warm. "
                     "Includes dataset loading/preparation, device transfer, forward + softmax, "
                     "and output to CPU. Excludes checkpoint loading, raw video decoding, "
                     "landmark extraction, temporal window collection and UI. Rates count "
                     "classified samples/windows, not camera frames. Synchronized, serial execution.",
        }

    def _predict(self, model, inputs):
        output = model(inputs)
        return output if getattr(model, "OUTPUT_FORMAT", "logits") == "probabilities" else output.softmax(dim=1)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _statistics(values):
        return {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "min": float(np.min(values)), "max": float(np.max(values)),
            "std": float(np.std(values)), "total": float(np.sum(values)),
        }
