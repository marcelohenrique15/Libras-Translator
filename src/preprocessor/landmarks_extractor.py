from pathlib import Path
import csv
import json
import tempfile
from hashlib import sha256

import cv2
import mediapipe as mp


class LandmarksExtractor():
    LANDMARK_GROUPS = (
        ("pose", 33),
        ("face", 468),
        ("left_hand", 21),
        ("right_hand", 21),
    )

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self.manifest_path = self.dataset_root / "metadata" / "manifest.csv"
        self.landmarks_dir = self.dataset_root / "processed" / "landmarks"
        self.index_path = self.landmarks_dir / "index.json"
        self.version = sha256(Path(__file__).read_bytes() + mp.__version__.encode()).hexdigest()[:12]
        index = json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else {}
        self.stale_version = "version" in index and index["version"] != self.version
        self.completed = index.get("samples", index) if not self.stale_version else {}
        self.pending = set(index.get("pending", []))
        if self.stale_version:
            self.pending = {path.stem for path in self.landmarks_dir.glob("*.csv")}

    #Público
    def extract(self) -> None:
        rows = self._read_manifest()
        self.landmarks_dir.mkdir(parents=True, exist_ok=True)
        reused = 0
        print("Conferindo os landmarks existentes...", flush=True)

        for index, row in enumerate(rows, start=1):
            video_path = self.dataset_root / row["path"]
            output_path = self.landmarks_dir / f"{row['sample_id']}.csv"
            if index % 100 == 0:
                print(f"Landmarks conferidos: {index}/{len(rows)}", flush=True)
            if self._is_complete(video_path, output_path):
                reused += 1
                if video_path.exists():
                    self.completed[row["sample_id"]] = self._signature(video_path, output_path)
                continue

            print(f"Extraindo {index}/{len(rows)}: {row['sample_id']}", flush=True)
            with tempfile.NamedTemporaryFile(dir=self.landmarks_dir, suffix=".csv.tmp", delete=False) as file:
                temporary_path = Path(file.name)
            self._extract_video(video_path, temporary_path)
            temporary_path.replace(output_path)
            self.completed[row["sample_id"]] = self._signature(video_path, output_path)
            self.pending.discard(row["sample_id"])
            self._save_index()

        self._save_index()
        print(f"Landmarks: {reused} reutilizados, {len(rows) - reused} extraídos.", flush=True)

    #Privado
    def _is_complete(self, video_path: Path, output_path: Path) -> bool:
        if not output_path.exists() or output_path.stem in self.pending:
            return False

        if not video_path.exists():
            stat = output_path.stat()
            recorded = self.completed.get(output_path.stem)
            if recorded is not None and recorded[-2:] == [stat.st_size, stat.st_mtime_ns]:
                return True
            raise FileNotFoundError(f"Vídeo ausente e CSV sem confirmação de extração completa: {video_path}")

        if output_path.stat().st_mtime_ns < video_path.stat().st_mtime_ns:
            return False

        if self.completed.get(output_path.stem) == self._signature(video_path, output_path):
            return True

        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        with output_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            columns = self._columns()
            if next(reader, None) != columns:
                return False
            count = 0
            for index, row in enumerate(reader):
                if len(row) != len(columns) or row[0] != str(index):
                    return False
                count += 1

        if count > 0 and count == frame_count:
            return True

        # Alguns vídeos informam uma quantidade de frames diferente da leitura real.
        capture = cv2.VideoCapture(str(video_path))
        decoded_count = 0
        while capture.grab():
            decoded_count += 1
        capture.release()
        return count > 0 and count == decoded_count

    def _signature(self, video_path: Path, output_path: Path) -> list[int]:
        video_stat, output_stat = video_path.stat(), output_path.stat()
        return [video_stat.st_size, video_stat.st_mtime_ns, output_stat.st_size, output_stat.st_mtime_ns]

    def _save_index(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", dir=self.landmarks_dir, suffix=".json.tmp", delete=False, encoding="utf-8") as file:
            json.dump({"version": self.version, "samples": self.completed, "pending": sorted(self.pending)}, file)
            temporary_path = Path(file.name)
        temporary_path.replace(self.index_path)

    def _read_manifest(self) -> list[dict[str, str]]:
        with self.manifest_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

        return rows

    def _extract_video(self, video_path: Path, output_path: Path) -> None:
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        try:
            with mp.solutions.holistic.Holistic(
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
                refine_face_landmarks=False,
            ) as holistic, output_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self._columns())

                frame_index = 0

                while True:
                    success, frame = capture.read()

                    if not success:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb_frame)
                    writer.writerow([frame_index, *self._landmarks(results)])
                    frame_index += 1
                if frame_index == 0:
                    raise ValueError(f"Nenhum frame foi lido: {video_path}")
        finally:
            capture.release()

    def _columns(self) -> list[str]:
        columns = ["frame"]

        for group, count in self.LANDMARK_GROUPS:
            for index in range(count):
                columns.extend((f"{group}_{index}_x", f"{group}_{index}_y"))

        return columns

    def _landmarks(self, results) -> list[float | None]:
        coordinates = []

        for group, count in self.LANDMARK_GROUPS:
            detected = getattr(results, f"{group}_landmarks")

            if detected is None:
                coordinates.extend([None, None] * count)

            else:
                for landmark in detected.landmark:
                    coordinates.extend((landmark.x, landmark.y))

        return coordinates
