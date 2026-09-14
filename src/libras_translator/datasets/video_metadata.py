import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


FFPROBE_FIELDS = (
    "stream=codec_type,codec_name,width,height,"
    "r_frame_rate,avg_frame_rate,nb_frames:"
    "format=duration"
)


class VideoProbeError(Exception):
    """Indica uma falha ao obter os metadados de um vídeo."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Armazena os metadados técnicos de um vídeo."""

    path: Path
    codec: str
    width: int
    height: int
    nominal_fps: float
    average_fps: float
    num_frames: int | None
    duration_seconds: float
    has_audio: bool


def probe_video(video_path: Path) -> VideoMetadata:
    """Lê os metadados de um vídeo com o ffprobe."""

    if not video_path.is_file():
        raise VideoProbeError(f"Vídeo não encontrado: {video_path}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        FFPROBE_FIELDS,
        "-of",
        "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    except FileNotFoundError as error:
        raise VideoProbeError("O executável ffprobe não foi encontrado.") from error
    except subprocess.TimeoutExpired as error:
        raise VideoProbeError(f"Tempo limite de 30s excedido: {video_path}") from error

    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "erro desconhecido"
        raise VideoProbeError(f"Falha ao inspecionar {video_path}: {message}") from error

    try:
        ffprobe_data = json.loads(result.stdout)

    except json.JSONDecodeError as error:
        raise VideoProbeError(f"O ffprobe retornou JSON inválido para {video_path}.") from error

    video_stream = None
    has_audio = False

    if not isinstance(ffprobe_data, dict):
        raise VideoProbeError(f"Resposta inesperada do ffprobe: {video_path}")
    streams = ffprobe_data.get("streams", [])
    if not isinstance(streams, list):
        raise VideoProbeError(f"Lista de streams inválida: {video_path}")
    for stream in streams:
        if not isinstance(stream, dict):
            raise VideoProbeError(f"Stream inválido: {video_path}")
        stream_type = stream.get("codec_type")

        if stream_type == "video" and video_stream is None:
            video_stream = stream
        elif stream_type == "audio":
            has_audio = True

    if video_stream is None:
        raise VideoProbeError(f"Nenhum stream de vídeo encontrado em {video_path}.")

    try:
        raw_num_frames = video_stream.get("nb_frames")
        num_frames = None
        if raw_num_frames not in (None, "N/A"):
            num_frames = int(raw_num_frames)

        metadata = VideoMetadata(
            path=video_path,
            codec=video_stream["codec_name"],
            width=int(video_stream["width"]),
            height=int(video_stream["height"]),
            nominal_fps=float(Fraction(video_stream["r_frame_rate"])),
            average_fps=float(Fraction(video_stream["avg_frame_rate"])),
            num_frames=num_frames,
            duration_seconds=float(ffprobe_data["format"]["duration"]),
            has_audio=has_audio,
        )
        positive_values = (
            metadata.width, metadata.height, metadata.nominal_fps,
            metadata.average_fps, metadata.duration_seconds,
        )
        for value in positive_values:
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Dimensões, FPS e duração devem ser positivos e finitos.")
        if num_frames is not None and num_frames <= 0:
            raise ValueError("Número de frames deve ser positivo quando informado.")
        return metadata

    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise VideoProbeError(f"Metadados ausentes ou inválidos em {video_path}.") from error
