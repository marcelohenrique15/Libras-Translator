from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from libras_translator.datasets.audit import VideoAuditFailure, VideoDatasetAuditor
from libras_translator.datasets.minds_libras.sample import (
    InvalidMindsLibrasFilename,
    MindsLibrasSample,
    parse_minds_libras_filename,
)
from libras_translator.datasets.video_metadata import VideoMetadata


@dataclass(frozen=True, slots=True)
class MindsLibrasRecord:
    """Reúne a identidade e os metadados técnicos de uma amostra."""

    sample: MindsLibrasSample
    video: VideoMetadata


@dataclass(frozen=True, slots=True)
class FilenameAuditFailure:
    """Representa um arquivo com nomenclatura inválida."""

    path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class MindsLibrasAudit:
    """Armazena o resultado completo da auditoria do MINDS-Libras."""

    records: tuple[MindsLibrasRecord, ...]
    video_failures: tuple[VideoAuditFailure, ...]
    filename_failures: tuple[FilenameAuditFailure, ...]

    @property
    def is_valid(self) -> bool:
        return bool(self.records) and self.invalid_samples == 0

    @property
    def total_files(self) -> int:
        return len(self.records) + len(self.video_failures) + len(self.filename_failures)

    @property
    def valid_samples(self) -> int:
        return len(self.records)

    @property
    def invalid_samples(self) -> int:
        return len(self.video_failures) + len(self.filename_failures)

    @property
    def class_counts(self) -> dict[tuple[int, str], int]:
        counts = {}

        for record in self.records:
            class_key = (
                record.sample.class_id,
                record.sample.label,
            )
            current_count = counts.get(class_key, 0)
            counts[class_key] = current_count + 1

        return dict(sorted(counts.items()))

    @property
    def signer_counts(self) -> dict[int, int]:
        counts = {}

        for record in self.records:
            signer_id = record.sample.signer_id
            current_count = counts.get(signer_id, 0)
            counts[signer_id] = current_count + 1

        return dict(sorted(counts.items()))

    @property
    def repetition_counts(self) -> dict[int, int]:
        counts = {}

        for record in self.records:
            repetition = record.sample.repetition
            current_count = counts.get(repetition, 0)
            counts[repetition] = current_count + 1

        return dict(sorted(counts.items()))


class MindsLibrasAuditor:
    """Executa as auditorias técnica e semântica do MINDS-Libras."""

    def __init__(self, raw_directory: Path) -> None:
        self.video_auditor = VideoDatasetAuditor(raw_directory)

    def run(self, progress: Callable[[int, int], None] | None = None) -> MindsLibrasAudit:
        video_audit = self.video_auditor.run(progress=progress)

        records = []
        filename_failures = []

        for video_metadata in video_audit.videos:
            try:
                sample = parse_minds_libras_filename(video_metadata.path)

                record = MindsLibrasRecord(
                    sample=sample,
                    video=video_metadata,
                )

                records.append(record)

            except InvalidMindsLibrasFilename as error:
                failure = FilenameAuditFailure(
                    path=video_metadata.path,
                    reason=str(error),
                )

                filename_failures.append(failure)

        return MindsLibrasAudit(
            records=tuple(records),
            video_failures=video_audit.failures,
            filename_failures=tuple(filename_failures),
        )
