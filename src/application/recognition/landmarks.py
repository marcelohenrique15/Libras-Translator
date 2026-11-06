"""MediaPipe pertence a uma única thread e recebe a imagem original da câmera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from application.types import PersonBox


@dataclass(frozen=True)
class LandmarkObservation:
    points: np.ndarray
    box: PersonBox | None
    hands_present: bool
    caption_anchor: tuple[float, float] | None = None


class LandmarkDetector:
    """Extrai exatamente os grupos e a ordem usados pelo pré-processamento."""

    def __init__(self, groups: tuple) -> None:
        import mediapipe as mp

        self.groups = groups
        self._holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            refine_face_landmarks=False,
            model_complexity=1,
        )

    def process(self, image: np.ndarray) -> LandmarkObservation:
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.setflags(write=False)
        detected = self._holistic.process(rgb)
        points = []
        for group, indices in self.groups:
            landmarks = getattr(detected, f"{group}_landmarks", None)
            for index in indices:
                if landmarks is None:
                    points.append((np.nan, np.nan))
                else:
                    point = landmarks.landmark[index]
                    points.append((point.x, point.y))
        coordinates = np.asarray(points, dtype=np.float32)
        coordinates[~np.isfinite(coordinates)] = np.nan
        hands_present = any(
            getattr(detected, f"{side}_hand_landmarks", None) is not None
            for side in ("left", "right")
        )
        return LandmarkObservation(
            coordinates, self._person_box(detected), hands_present,
            self._chest_anchor(detected, image.shape),
        )

    def close(self) -> None:
        self._holistic.close()

    def reset(self) -> None:
        """Esquece o rastreamento da pessoa anterior, mantendo os recursos abertos."""
        self._holistic.reset()

    @staticmethod
    def _chest_anchor(detected: object, image_shape: tuple) -> tuple[float, float] | None:
        """Centro do peito: abaixo dos ombros, proporcional à largura do tronco."""
        pose = getattr(detected, "pose_landmarks", None)
        if pose is None or len(pose.landmark) <= 12:
            return None
        shoulders = [pose.landmark[11], pose.landmark[12]]
        if any(getattr(point, "visibility", 1) < 0.4 for point in shoulders):
            return None
        points = np.array([(point.x, point.y) for point in shoulders])
        if not np.isfinite(points).all():
            return None
        height, width = image_shape[:2]
        shoulder_width = np.linalg.norm((points[1] - points[0]) * [width, height])
        if shoulder_width < 1:
            return None
        center_x, center_y = points.mean(axis=0)
        return float(center_x), float(center_y + 0.25 * shoulder_width / height)

    @staticmethod
    def _person_box(detected: object) -> PersonBox | None:
        """Caixa do tronco/rosto; pontos fora do enquadramento são limitados."""
        coordinates = []
        pose = getattr(detected, "pose_landmarks", None)
        face = getattr(detected, "face_landmarks", None)
        if pose is not None:
            coordinates.extend(
                (point.x, point.y) for point in pose.landmark[:25]
                if getattr(point, "visibility", 1) >= 0.4
            )
        if face is not None:
            coordinates.extend((point.x, point.y) for point in face.landmark)
        if not coordinates:
            return None
        points = np.asarray(coordinates)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 2:
            return None
        left, top = np.clip(points.min(axis=0) - 0.025, 0, 1)
        right, bottom = np.clip(points.max(axis=0) + 0.025, 0, 1)
        if right - left < 0.02 or bottom - top < 0.02:
            return None
        return PersonBox(float(left), float(top), float(right), float(bottom))
