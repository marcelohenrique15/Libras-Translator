import json
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class LandmarkProcessor():
    FACE_POINTS = (
        40, 37, 267, 270, 291, 181, 17, 405, 375,
        80, 82, 312, 310, 308, 88, 87, 317, 318,
    )
    POSE_POINTS = (
        0, 1, 3, 4, 5, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    )
    LANDMARK_GROUPS = (
        ("face", FACE_POINTS),
        ("left_hand", tuple(range(21))),
        ("pose", POSE_POINTS),
        ("right_hand", tuple(range(21))),
    )
    SUBSETS = {
        "asl_2nd": LANDMARK_GROUPS,
        "all": (
            ("face", tuple(range(468))),
            ("left_hand", tuple(range(21))),
            ("pose", tuple(range(33))),
            ("right_hand", tuple(range(21))),
        ),
        "arcanjo": (
            ("pose", tuple(range(33))),
            ("left_hand", tuple(range(21))),
            ("right_hand", tuple(range(21))),
        ),
    }

    def __init__(
        self,
        dataset_root: Path,
        subset: str = "asl_2nd",
        imputation: bool = True,
        representation: str = "image",
        frame_count: int = 64,
        image_size: int = 224,
        augmentation: bool = True,
    ) -> None:
        if subset not in self.SUBSETS:
            raise ValueError(f"Subset desconhecido: {subset}. Use: {', '.join(self.SUBSETS)}.")
        if representation not in {"image", "sequence"}:
            raise ValueError("Use representation='image' ou 'sequence'.")
        if frame_count < 1 or image_size < 1:
            raise ValueError("frame_count e image_size devem ser positivos.")

        self.landmarks_dir = dataset_root / "processed" / "landmarks"
        self.landmark_groups = self.SUBSETS[subset]
        self.landmark_count = sum(len(points) for _, points in self.landmark_groups)
        self.input_size = self.landmark_count * 2
        self.imputation = imputation
        self.representation = representation
        self.frame_count = frame_count
        self.image_size = image_size
        self.augmentation = augmentation

        code_version = sha256(Path(__file__).read_bytes()).hexdigest()
        selection_options = {"code": code_version, "subset": subset}
        imputation_options = {**selection_options, "interpolation_limit": 5}
        encoding_options = {
            **selection_options,
            "imputation": imputation,
            "representation": representation,
            "size": image_size if representation == "image" else frame_count,
        }
        processed_dir = dataset_root / "processed"
        self.selection_dir = processed_dir / "selected_landmarks" / subset / self._version(selection_options)
        self.imputation_dir = processed_dir / "imputed_landmarks" / subset / self._version(imputation_options)
        self.encoding_dir = processed_dir / "encoded_inputs" / representation / self._version(encoding_options)
        self.cache_dir = self.imputation_dir if imputation else self.selection_dir

    # Público
    def select(self, sample_id: str) -> np.ndarray:
        csv_path = self.landmarks_dir / f"{sample_id}.csv"
        cache_path = self.selection_dir / f"{sample_id}.npy"
        if self._is_current(cache_path, csv_path):
            return np.load(cache_path, allow_pickle=False)

        table = self._read_landmarks(csv_path)
        landmarks = table.to_numpy(dtype=np.float32).reshape(-1, self.landmark_count, 2)
        self._save_array(cache_path, landmarks)
        return landmarks

    def impute(self, sample_id: str) -> np.ndarray:
        csv_path = self.landmarks_dir / f"{sample_id}.csv"
        cache_path = self.imputation_dir / f"{sample_id}.npy"
        if self._is_current(cache_path, csv_path):
            return np.load(cache_path, allow_pickle=False)

        selected = self.select(sample_id)
        table = pd.DataFrame(selected.reshape(len(selected), -1), columns=self._columns())
        landmarks = self._impute(table)
        self._save_array(cache_path, landmarks)
        return landmarks

    def prepare(self, sample_id: str) -> np.ndarray:
        if self.imputation:
            return self.impute(sample_id)
        return np.nan_to_num(self.select(sample_id), nan=0.0)

    def encode(self, landmarks: np.ndarray) -> np.ndarray:
        if self.representation == "sequence":
            return self._encode_sequence(landmarks)
        return self._encode(landmarks)

    def process(self, sample_id: str) -> np.ndarray:
        csv_path = self.landmarks_dir / f"{sample_id}.csv"
        cache_path = self.encoding_dir / f"{sample_id}.npy"
        if self._is_current(cache_path, csv_path):
            encoded = np.load(cache_path, allow_pickle=False)
            if encoded.dtype == np.uint8:
                return encoded.astype(np.float32) / 255
            return encoded

        encoded = self.encode(self.prepare(sample_id))
        # A imagem já usa valores de 8 bits; o cache menor preserva esses valores.
        cached = np.rint(encoded * 255).astype(np.uint8) if self.representation == "image" else encoded
        self._save_array(cache_path, cached)
        return encoded

    def augment(self, landmarks: np.ndarray) -> np.ndarray:
        if not self.augmentation:
            return landmarks

        landmarks = np.clip(landmarks, 0, 1).copy()
        center = landmarks.mean(axis=(0, 1), keepdims=True)
        angle = np.deg2rad(np.random.normal(0, 12))
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ], dtype=np.float32)

        landmarks = (landmarks - center) @ rotation.T + center
        landmarks *= np.random.normal(1, 0.1)
        landmarks[:, :, 0] += np.random.normal(0, 0.06)
        if np.random.random() < 0.5:
            landmarks[:, :, 0] = 1 - landmarks[:, :, 0]
        return landmarks

    def prepare_all(self, rows: list[dict[str, str]], stage: str = "encode") -> None:
        if stage not in {"select", "impute", "encode"}:
            raise ValueError("Use stage='select', 'impute' ou 'encode'.")

        steps = [("Seleção dos pontos", self.select, self.selection_dir)]
        if stage == "impute" or (stage == "encode" and self.imputation):
            steps.append(("Imputação das coordenadas", self.impute, self.imputation_dir))
        if stage == "encode":
            steps.append((f"Codificação como {self.representation}", self.process, self.encoding_dir))

        for description, operation, cache_dir in steps:
            print(f"{description}...", flush=True)
            reused = 0
            for index, row in enumerate(rows, start=1):
                sample_id = row["sample_id"]
                csv_path = self.landmarks_dir / f"{sample_id}.csv"
                if self._is_current(cache_dir / f"{sample_id}.npy", csv_path):
                    reused += 1
                operation(sample_id)
                if index % 100 == 0:
                    print(f"Amostras preparadas: {index}/{len(rows)}", flush=True)
            print(f"{description}: {reused} reutilizadas, {len(rows) - reused} processadas.", flush=True)

    # Privado
    def _columns(self) -> list[str]:
        columns = []
        for group, points in self.landmark_groups:
            for point in points:
                columns.extend((f"{group}_{point}_x", f"{group}_{point}_y"))
        return columns

    def _read_landmarks(self, csv_path: Path) -> pd.DataFrame:
        columns = self._columns()
        landmarks = pd.read_csv(csv_path, usecols=columns, dtype=np.float32)
        return landmarks[columns]

    def _impute(self, landmarks: pd.DataFrame) -> np.ndarray:
        for column in landmarks.columns:
            method = "cubic" if landmarks[column].count() >= 4 else "linear"
            landmarks[column] = landmarks[column].interpolate(
                method=method,
                limit=5,
                limit_direction="both",
            )
        return landmarks.fillna(0).to_numpy(dtype=np.float32).reshape(-1, self.landmark_count, 2)

    def _encode(self, landmarks: np.ndarray) -> np.ndarray:
        if len(landmarks) < 3:
            raise ValueError("A codificação como imagem precisa de pelo menos 3 frames.")
        frame_count = len(landmarks) // 3 * 3
        point_count = landmarks.shape[1]
        x = landmarks[:frame_count, :, 0].T.reshape(point_count, -1, 3)
        y = landmarks[:frame_count, :, 1].T.reshape(point_count, -1, 3)
        image = np.concatenate((x, y), axis=1)
        image = np.uint8(np.clip(image, 0, 1) * 255)
        image = Image.fromarray(image).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR,
        )
        return np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1), dtype=np.float32) / 255

    def _encode_sequence(self, landmarks: np.ndarray) -> np.ndarray:
        if len(landmarks) == 0:
            raise ValueError("A codificação como sequência precisa de pelo menos 1 frame.")
        coordinates = landmarks.reshape(len(landmarks), -1)
        source_time = np.linspace(0, 1, len(landmarks))
        target_time = np.linspace(0, 1, self.frame_count)
        sequence = np.empty((self.frame_count, coordinates.shape[1]), dtype=np.float32)
        for column in range(coordinates.shape[1]):
            sequence[:, column] = np.interp(target_time, source_time, coordinates[:, column])
        return sequence

    def _version(self, options: dict) -> str:
        return sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:12]

    def _is_current(self, cache_path: Path, csv_path: Path) -> bool:
        return cache_path.exists() and cache_path.stat().st_mtime_ns >= csv_path.stat().st_mtime_ns

    def _save_array(self, cache_path: Path, values: np.ndarray) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npy.tmp", delete=False) as file:
            temporary_path = Path(file.name)
            np.save(file, values)
        temporary_path.replace(cache_path)
