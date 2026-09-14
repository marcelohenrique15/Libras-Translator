"""Registro dos comandos públicos do projeto."""

import argparse
import sys
from pathlib import Path

from libras_translator.cli.audit import run_audit
from libras_translator.cli.minds_libras import run_manifest


MANIFEST_HANDLERS = {"minds-libras": run_manifest}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="libras-translator",
        description="Ferramentas para experimentos de reconhecimento de Libras.",
    )
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument(
        "--audit", type=Path, metavar="PASTA",
        help="Inspeciona os vídeos diretamente nesta pasta, sem depender do dataset.",
    )
    commands.add_argument(
        "--build-manifest", type=Path, metavar="RAIZ_DATASET",
        help="Audita raw/ dentro desta raiz e grava o manifesto CSV.",
    )
    parser.add_argument(
        "--dataset", choices=MANIFEST_HANDLERS,
        help="Formato dos rótulos, somente para manifesto (padrão: minds-libras).",
    )
    parser.add_argument("--output", type=Path, help="Destino do manifesto CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Autoriza substituir o manifesto.")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = create_parser().parse_args(argv)
    try:
        if arguments.audit is not None:
            if arguments.output is not None or arguments.overwrite or arguments.dataset is not None:
                raise ValueError("--dataset, --output e --overwrite são opções de --build-manifest.")
            return run_audit(arguments.audit)
        dataset = arguments.dataset or "minds-libras"
        return MANIFEST_HANDLERS[dataset](arguments)
    except (ValueError, OSError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nOperação interrompida.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
