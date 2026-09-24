from pathlib import Path
import csv


class LandmarksExtractor():
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.manifest_path = self.dataset_root / "metadata" / "manifest.csv"
        self.landmarks_dir = self.dataset_root / "processed" / "landmarks"
        self.raw_data_dir = self.dataset_root / "raw"

    #Público
    def extract(self) -> None:
        pass

    #Privado
    def _read_manifest(self) -> list[dict[str, str]]:
        with self.manifest_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        return rows