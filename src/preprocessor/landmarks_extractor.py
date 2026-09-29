from pathlib import Path
import csv

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

    #Público
    def extract(self) -> None:
        rows = self._read_manifest()
        self.landmarks_dir.mkdir(parents=True, exist_ok=True)

        for index, row in enumerate(rows, start=1):
            video_path = self.dataset_root / row["path"]
            output_path = self.landmarks_dir / f"{row['sample_id']}.csv"
            print(f"Extraindo {index}/{len(rows)}: {row['sample_id']}", flush=True)
            self._extract_video(video_path, output_path)

    #Privado
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
