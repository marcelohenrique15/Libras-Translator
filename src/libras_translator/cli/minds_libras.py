"""Argumentos e apresentação dos comandos do MINDS-Libras."""

import argparse
from libras_translator.cli.audit import show_progress
from libras_translator.datasets.minds_libras.audit import MindsLibrasAudit
from libras_translator.datasets.minds_libras.manifest import build_manifest


def print_report(report: MindsLibrasAudit) -> None:
    print(f"Arquivos examinados: {report.total_files}")
    print(f"Metadados e nomes legíveis: {report.valid_samples}")
    print(f"Falhas por arquivo: {report.invalid_samples}")
    print("Vídeos por classe:")
    for (class_id, label), count in report.class_counts.items():
        print(f"  {class_id:02d} {label}: {count}")
    print(f"Vídeos por sinalizador: {report.signer_counts}")
    print(f"Vídeos por repetição: {report.repetition_counts}")
    for failure in report.video_failures + report.filename_failures:
        print(f"  {failure.path}: {failure.reason}")
    print("Auditoria aprovada." if report.is_valid else "Auditoria reprovada.")


def run_manifest(arguments: argparse.Namespace) -> int:
    dataset_root = arguments.build_manifest
    output = arguments.output or dataset_root / "metadata" / "manifest.csv"
    print(f"Formato dos rótulos: minds-libras. Auditando: {dataset_root / 'raw'}", flush=True)
    report = build_manifest(
        dataset_root, output,
        overwrite=arguments.overwrite, progress=show_progress,
    )
    print_report(report)
    if not report.is_valid:
        print("Manifesto não gravado; corrija os problemas indicados.")
        return 1
    print(f"Manifesto gravado: {output}")
    return 0

