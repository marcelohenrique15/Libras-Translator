import argparse
from pathlib import Path

from dataset.manifest_builder import ManifestBuilder
from dataset.manifest_splitter import ManifestSplitter


def main() -> None:
    parser = argparse.ArgumentParser(prog="libras-translator")
    parser.add_argument("--build-manifest", type=Path, metavar="DATASET_ROOT")

    args = parser.parse_args()

    if args.build_manifest is not None:
        manifest_path = ManifestBuilder(args.build_manifest).build()
        ManifestSplitter().split(manifest_path, "05")
        print(f"Manifesto criado em: {manifest_path}")


if __name__ == "__main__":
    main()
