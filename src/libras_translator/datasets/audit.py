from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from libras_translator.datasets.video_metadata import (
    VideoMetadata,
    VideoProbeError,
    probe_video,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True, slots=True)
class VideoAuditFailure:
    """Representa um vídeo que não pôde ser inspecionado."""

    path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class VideoDatasetAudit:
    """Armazena o resultado da auditoria técnica dos vídeos."""

    videos: tuple[VideoMetadata, ...]
    failures: tuple[VideoAuditFailure, ...]

    @property
    def total_files(self) -> int:
        return len(self.videos) + len(self.failures)

    @property
    def valid_files(self) -> int:
        return len(self.videos)

    @property
    def invalid_files(self) -> int:
        return len(self.failures)


class VideoDatasetAuditor:
    """Inspeciona tecnicamente os vídeos encontrados em um diretório."""

    def __init__(self, video_directory: Path) -> None:
        self.video_directory = video_directory

    def run(self, progress: Callable[[int, int], None] | None = None) -> VideoDatasetAudit:
        self._validate_directory()
        paths = self._find_videos()
        if not paths:
            raise ValueError(f"Nenhum vídeo encontrado diretamente em {self.video_directory}")
        if which("ffprobe") is None:
            raise ValueError("Instale FFmpeg: o executável ffprobe não foi encontrado.")

        videos = []
        failures = []

        for index, video_path in enumerate(paths, start=1):
            try:
                metadata = probe_video(video_path)
                videos.append(metadata)

            except VideoProbeError as error:
                failure = VideoAuditFailure(
                    path=video_path,
                    reason=str(error),
                )

                failures.append(failure)
            if progress is not None:
                progress(index, len(paths))

        return VideoDatasetAudit(
            videos=tuple(videos),
            failures=tuple(failures),
        )

    def _validate_directory(self) -> None:
        if not self.video_directory.is_dir():
            raise ValueError(f"Diretório de dados não encontrado: {self.video_directory}")

    def _find_videos(self) -> list[Path]:
        videos = []

        for path in self.video_directory.iterdir():
            is_video = path.suffix.lower() in VIDEO_EXTENSIONS

            if path.is_file() and is_video:
                videos.append(path)

        return sorted(videos)
