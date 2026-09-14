import re
from dataclasses import dataclass
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^(?P<class_id>\d{2})"
    r"(?P<label>.+)"
    r"Sinalizador(?P<signer_id>\d{2})"
    r"-(?P<repetition>\d+)$"
)


class InvalidMindsLibrasFilename(ValueError):
    """Indica que um arquivo não segue a nomenclatura do MINDS-Libras."""


@dataclass(frozen=True, slots=True)
class MindsLibrasSample:
    """Identifica semanticamente uma amostra do MINDS-Libras."""

    path: Path
    class_id: int
    label: str
    signer_id: int
    repetition: int

    @property
    def sample_id(self) -> str:
        return self.path.stem


def parse_minds_libras_filename(video_path: Path) -> MindsLibrasSample:
    """Extrai as informações contidas no nome de um vídeo."""

    match = FILENAME_PATTERN.fullmatch(video_path.stem)

    if match is None:
        raise InvalidMindsLibrasFilename(
            f"Nome de arquivo inválido: {video_path.name}"
        )

    return MindsLibrasSample(
        path=video_path,
        class_id=int(match.group("class_id")),
        label=match.group("label"),
        signer_id=int(match.group("signer_id")),
        repetition=int(match.group("repetition")),
    )