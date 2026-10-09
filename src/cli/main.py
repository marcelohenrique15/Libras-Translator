import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="libras-translator")
    commands = parser.add_mutually_exclusive_group(required=True)
    for command in ("build-manifest", "split-manifest", "extract-landmarks", "select-landmarks", "impute-landmarks", "encode-landmarks", "train"):
        commands.add_argument(f"--{command}", nargs="?", const=True, type=Path, metavar="DATASET_ROOT")
    commands.add_argument("--list-models", action="store_true")

    parser.add_argument("--config", type=Path, help="Configuração TOML ou JSON")
    parser.add_argument("--dataset", type=Path, help="Nome em data/ ou caminho do dataset")
    parser.add_argument("--dataset-format", choices=("auto", "minds_libras", "class_signer"))
    parser.add_argument("--model", help="Modelo registrado ou pacote.modulo:Classe")
    parser.add_argument("--processor", help="Processador registrado ou pacote.modulo:Classe")
    parser.add_argument("--test-signer-id", help="ID de teste ou all para LOPO completo")
    parser.add_argument("--validation-signer-id", help="Executa apenas uma rodada de validação")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--final-epochs", type=int, help="Duração final já selecionada; dispensa validações internas sem busca")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--l1-lambda", type=float)
    parser.add_argument("--l2-lambda", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda ou cuda:N")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--search-config", type=Path, help="Espaço de busca TOML para o Optuna")
    parser.add_argument("--search-trials", type=int, help="Total de tentativas concluídas por estudo Optuna")
    parser.add_argument("--reuse-search", type=Path, help="Importa a configuração vencedora de uma busca salva")
    parser.add_argument("--force-restart", action="store_true", default=None, help="Cria nova execução, sem reutilizar treinos nem apagar resultados anteriores")
    parser.add_argument("--subset", choices=("asl_2nd", "all", "arcanjo"))
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--anchor", choices=("shoulders", "nose", "none"),
                        help="Referência fixa por vídeo: centro dos ombros, nariz ou sem ancoragem")
    parser.add_argument("--imputation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=None,
                        help="Compatibilidade: este experimento aceita somente --no-augmentation")
    args = parser.parse_args()

    if args.list_models:
        from models.model_registry import ModelRegistry

        for name in ModelRegistry.names():
            print(f"{name}: entrada {ModelRegistry.input_format(name)}")
        return

    from training.config import TrainingConfig

    config = TrainingConfig()
    if args.config is not None:
        if args.config.suffix == ".json":
            config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        else:
            config = TrainingConfig.from_toml(args.config)

    argument_names = (
        "dataset", "dataset_format", "model", "processor", "test_signer_id", "validation_signer_id",
        "epochs", "final_epochs", "batch_size", "learning_rate", "weight_decay", "patience", "device", "num_workers",
        "l1_lambda", "l2_lambda", "label_smoothing", "gradient_clip",
        "seed", "output_dir", "weights_dir", "search_config", "search_trials", "reuse_search", "force_restart",
    )
    overrides = {name: getattr(args, name) for name in argument_names if getattr(args, name) is not None}
    actions = ("build_manifest", "split_manifest", "extract_landmarks", "select_landmarks", "impute_landmarks", "encode_landmarks", "train")
    action = next(name for name in actions if getattr(args, name) is not None)
    if isinstance(getattr(args, action), Path):
        if args.dataset is not None:
            parser.error("Informe o dataset no comando ou em --dataset, uma única vez.")
        overrides["dataset"] = getattr(args, action)
    processor_options = dict(config.processor_options)
    for name in ("subset", "frame_count", "anchor", "imputation", "augmentation"):
        if getattr(args, name) is not None:
            processor_options[name] = getattr(args, name)
    overrides["processor_options"] = processor_options
    config = config.with_overrides(overrides)
    if len(config.dataset.parts) == 1 and not config.dataset.exists():
        config = config.with_overrides({"dataset": Path("data") / config.dataset})

    if action == "build_manifest":
        from dataset.manifest_builder import ManifestBuilder

        manifest_path = ManifestBuilder(config.dataset, config.dataset_format).build()
        if config.test_signer_id not in {None, "all"}:
            _split_manifest(manifest_path, config.test_signer_id)
        print(f"Manifesto criado em: {manifest_path}")
        return

    if action == "split_manifest":
        if config.test_signer_id in {None, "all"}:
            parser.error("--split-manifest exige --test-signer-id com um ID.")
        _split_manifest(config.dataset / "metadata" / "manifest.csv", config.test_signer_id)
        return

    if action == "extract_landmarks":
        from dataset.manifest_builder import ManifestBuilder
        from preprocessor.landmarks_extractor import LandmarksExtractor

        if not (config.dataset / "metadata" / "manifest.csv").exists():
            ManifestBuilder(config.dataset, config.dataset_format).build()
        LandmarksExtractor(config.dataset).extract()
        return

    from training.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline(config)
    if action == "train":
        pipeline.train()
    else:
        stage = {"select_landmarks": "select", "impute_landmarks": "impute", "encode_landmarks": "encode"}[action]
        pipeline.prepare(stage)


def _split_manifest(manifest_path: Path, signer_id: str) -> None:
    import csv
    from dataset.manifest_splitter import ManifestSplitter

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        signers = {row["signer_id"] for row in csv.DictReader(file)}
    if signer_id not in signers and signer_id.zfill(2) in signers:
        signer_id = signer_id.zfill(2)
    if signer_id not in signers:
        raise ValueError(f"Sinalizador ausente no manifesto: {signer_id}")
    ManifestSplitter().split(manifest_path, signer_id)
    print(f"Divisão salva em: {manifest_path}")


if __name__ == "__main__":
    main()
