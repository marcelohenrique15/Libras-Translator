"""Apresentação da auditoria técnica de qualquer pasta de vídeos."""

from pathlib import Path

from libras_translator.datasets.audit import VideoDatasetAuditor


def show_progress(done: int, total: int) -> None:
    if done == 1 or done % 50 == 0 or done == total:
        print(f"Inspecionados: {done}/{total}", flush=True)


def run_audit(directory: Path) -> int:
    print(f"Auditando vídeos em: {directory}", flush=True)
    report = VideoDatasetAuditor(directory).run(progress=show_progress)
    print(f"Arquivos examinados: {report.total_files}")
    print(f"Metadados legíveis: {report.valid_files}")
    print(f"Falhas por arquivo: {report.invalid_files}")
    for failure in report.failures:
        print(f"  {failure.path}: {failure.reason}")
    print("Auditoria aprovada." if not report.failures else "Auditoria reprovada.")
    return 1 if report.failures else 0
