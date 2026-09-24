from pathlib import Path
import csv



class ManifestBuilder():
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_dir = dataset_root
        self.raw_dir = self.dataset_dir / "raw"

    # Público
    def build(self) -> Path:
        metadata_dir = self.dataset_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = metadata_dir / "manifest.csv"
        manifest = self._manifest_rows()

        with manifest_path.open("w", encoding="utf-8", newline="") as file:
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

        return manifest_path

    # Privado
    def _parse_video_name(self, video_path: Path) -> tuple[str, str, str, str, str]:
        sample_id = video_path.stem
        id_label, signer_repetition = sample_id.split("Sinalizador")

        class_id = id_label[:2]
        label = id_label[2:]
        signer, repetition = signer_repetition.split("-")

        return sample_id, class_id, label, signer, repetition

    def _videos(self) -> list[Path]:
            videos = list(self.raw_dir.rglob("*.mp4"))
            return videos
    
    def _manifest_rows(self) -> list:
        videos = []
        videos_paths = self._videos()

        for path in videos_paths:
            sample_id, class_id, label, signer_id, repetition = self._parse_video_name(path)

            video = {
                "sample_id": sample_id,
                "class_id": class_id,
                "label": label,
                "signer_id": signer_id,
                "repetition": repetition,
                "path": path.relative_to(self.dataset_dir).as_posix()
            }

            videos.append(video)

        return videos