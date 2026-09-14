"""Contrato e persistência do manifesto de vídeos de sinais isolados."""

import csv
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """Uma amostra; path é POSIX relativo à raiz do dataset, nunca ao CSV."""

    sample_id: str
    path: str
    class_id: int
    label: str
    signer_id: int
    repetition: int
    codec: str
    width: int
    height: int
    nominal_fps: float
    average_fps: float
    num_frames: int | None
    duration_seconds: float
    has_audio: bool


class ManifestWriter:
    """Publica um CSV completo, sem deixar um manifesto parcialmente escrito."""

    def write(
        self, rows: Sequence[ManifestRow], output: Path, *, overwrite: bool = False,
    ) -> None:
        if not rows:
            raise ValueError("Não é possível escrever um manifesto vazio.")
        if output.exists() and not overwrite:
            raise FileExistsError(f"O manifesto já existe: {output}. Use --overwrite.")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=output.parent,
                prefix=f".{output.name}.", suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                writer = csv.DictWriter(
                    temporary, fieldnames=[field.name for field in fields(ManifestRow)],
                )
                writer.writeheader()
                for row in sorted(rows, key=lambda row: row.sample_id):
                    writer.writerow(asdict(row))
                temporary.flush()
                os.fsync(temporary.fileno())
            if overwrite:
                os.replace(temporary_path, output)
            else:
                # Não sobrescreve nem se outro processo criar o destino durante a escrita.
                os.link(temporary_path, output)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
