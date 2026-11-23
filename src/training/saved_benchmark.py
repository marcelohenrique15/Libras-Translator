"""Mede um teste já concluído, sem executar busca ou treinamento."""

import csv
import inspect
import json
from pathlib import Path

import torch

from dataset.sign_dataset import SignDataset
from models.model_registry import ModelRegistry
from preprocessor.processor_registry import ProcessorRegistry
from training.reporting import ExperimentReporter
from training.trainer import Trainer


def benchmark_saved_run(run_dir: Path, dataset_root: Path | None = None, device: str = "auto") -> dict:
    run_dir = Path(run_dir)
    test_path = run_dir / "test.json"
    if not test_path.is_file():
        raise ValueError("Informe a pasta final do teste, contendo test.json e final.pt ou best.pt.")
    test = json.loads(test_path.read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "final.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    settings = dict(checkpoint["config"])
    mapping = checkpoint["class_to_index"]
    root = Path(dataset_root if dataset_root is not None else settings["dataset"])
    settings["dataset"] = str(root.resolve())
    with (root / "metadata/manifest.csv").open(encoding="utf-8", newline="") as file:
        manifest = {row["sample_id"]: row for row in csv.DictReader(file)}
    missing = set(test["sample_ids"]) - manifest.keys()
    if missing:
        raise ValueError(f"O manifesto não contém as amostras do teste: {sorted(missing)}")
    rows = [manifest[sample_id] for sample_id in test["sample_ids"]]
    processor = ProcessorRegistry.create(settings["processor"], root, settings["processor_options"])
    options = dict(settings["model_options"])
    if "pretrained" in inspect.signature(ModelRegistry.get_class(settings["model"])).parameters:
        options["pretrained"] = False
    model = ModelRegistry.create(settings["model"], len(mapping), options)
    settings["device"] = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    trainer = Trainer(run_dir, mapping, settings)
    result = trainer.test(model, SignDataset(rows, processor, mapping))
    result_path = run_dir / "result.json"
    if result_path.exists():
        report = json.loads(result_path.read_text(encoding="utf-8"))
        report["performance"] = result["performance"]
        trainer.write_json("result.json", report)
    labels = [key for key in sorted(mapping, key=mapping.get)]
    reporter = ExperimentReporter(run_dir, labels)
    (run_dir / "graphs").mkdir(exist_ok=True)
    reporter.write_performance(result["performance"])
    print(f"Desempenho salvo em: {run_dir.resolve() / 'performance.json'}", flush=True)
    return result["performance"]
