"""Conversão e caso de uso para construir o manifesto do MINDS-Libras."""

from collections.abc import Callable
from pathlib import Path

from libras_translator.datasets.manifest import ManifestRow, ManifestWriter
from libras_translator.datasets.minds_libras.audit import MindsLibrasAudit, MindsLibrasAuditor


def manifest_rows(report: MindsLibrasAudit, dataset_root: Path) -> tuple[ManifestRow, ...]:
    """Adapta o relatório ao contrato comum, mantendo os caminhos portáveis."""
    if not report.is_valid:
        raise ValueError("A auditoria precisa ser aprovada antes de gerar o manifesto.")
    root = dataset_root.resolve()
    rows = []
    for record in report.records:
        sample = record.sample
        video = record.video
        relative_path = sample.path.resolve().relative_to(root)
        rows.append(ManifestRow(
            sample_id=sample.sample_id, path=relative_path.as_posix(),
            class_id=sample.class_id, label=sample.label,
            signer_id=sample.signer_id, repetition=sample.repetition,
            codec=video.codec, width=video.width, height=video.height,
            nominal_fps=video.nominal_fps, average_fps=video.average_fps,
            num_frames=video.num_frames, duration_seconds=video.duration_seconds,
            has_audio=video.has_audio,
        ))
    return tuple(rows)


def build_manifest(
    dataset_root: Path, output: Path | None = None, *,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> MindsLibrasAudit:
    """Audita uma única vez e grava somente se todas as verificações passarem."""
    root = dataset_root.resolve()
    destination = output if output is not None else root / "metadata" / "manifest.csv"
    destination = destination.resolve()
    raw = root / "raw"
    if destination.is_relative_to(raw.resolve()):
        raise ValueError("O manifesto não pode ser gravado dentro de raw/.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"O manifesto já existe: {destination}. Use --overwrite.")
    report = MindsLibrasAuditor(raw).run(progress=progress)
    if report.is_valid:
        ManifestWriter().write(manifest_rows(report, root), destination, overwrite=overwrite)
    return report
