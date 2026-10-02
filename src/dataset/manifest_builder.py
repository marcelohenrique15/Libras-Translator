from pathlib import Path
import csv
import re
import tempfile

class ManifestBuilder():
    def __init__(self, dataset_root: Path, dataset_format: str = "auto") -> None:
        self.dataset_dir = dataset_root
        self.raw_dir = self.dataset_dir / "raw"
        self.dataset_format = dataset_format

        if dataset_format not in {"auto", "minds_libras", "class_signer"}:
            raise ValueError(f"Formato de dataset desconhecido: {dataset_format}")

    # Público
    def build(self) -> Path:
        metadata_dir = self.dataset_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = metadata_dir / "manifest.csv"
        manifest = self._manifest_rows()
        with tempfile.NamedTemporaryFile(mode="w", dir=metadata_dir, suffix=".csv.tmp", delete=False, encoding="utf-8", newline="") as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "sample_id",
                    "class_id",
                    "label",
                    "signer_id",
                    "repetition",
                    "path"
                ]
            )
            writer.writeheader()
            writer.writerows(manifest)

        temporary_path.replace(manifest_path)
        return manifest_path

    # Privado
    def _parse_minds_video(self, video_path: Path) -> dict[str, str] | None:
        sample_id = video_path.stem
        id_label, signer_repetition = sample_id.split("Sinalizador")
        class_id = id_label[:2]
        label = id_label[2:]
        signer_id, repetition = signer_repetition.split("-")

        if repetition not in {"1", "2", "3", "4", "5"}:
            return None

        return {
            "sample_id": sample_id,
            "class_id": class_id,
            "label": label,
            "signer_id": signer_id,
            "repetition": repetition,
            "path": video_path.relative_to(self.dataset_dir).as_posix(),
        }

    def _parse_class_signer_video(self, video_path: Path) -> dict[str, str]:
        relative_path = video_path.relative_to(self.raw_dir)
        class_id, signer_id = relative_path.parts[:2]

        return {
            "sample_id": "__".join(relative_path.with_suffix("").parts),
            "class_id": class_id,
            "label": class_id,
            "signer_id": signer_id,
            "repetition": video_path.stem,
            "path": video_path.relative_to(self.dataset_dir).as_posix(),
        }

    def _videos(self) -> list[Path]:
        return sorted(self.raw_dir.rglob("*.mp4"))

    def _resolve_format(self, videos: list[Path]) -> str:
        if self.dataset_format != "auto":
            return self.dataset_format

        if videos and re.fullmatch(r"\d{2}.+Sinalizador[^-]+-\d+", videos[0].stem):
            return "minds_libras"

        return "class_signer"

    def _manifest_rows(self) -> list[dict[str, str]]:
        videos = self._videos()
        dataset_format = self._resolve_format(videos)
        rows = []

        for video_path in videos:
            if dataset_format == "minds_libras":
                row = self._parse_minds_video(video_path)
            else:
                row = self._parse_class_signer_video(video_path)

            if row is not None:
                rows.append(row)

        return rows
