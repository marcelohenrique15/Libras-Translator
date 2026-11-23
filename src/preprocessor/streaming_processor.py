"""Entrada em memória para a aplicação, preservando o processor do experimento."""

import numpy as np
import pandas as pd

from preprocessor.landmark_processor import LandmarkProcessor


class StreamingLandmarkProcessor(LandmarkProcessor):
    """Adapta a origem dos dados; interpolação, ancoragem e encoder são herdados."""

    def prepare_array(self, landmarks: np.ndarray) -> np.ndarray:
        """Prepara uma janela já selecionada sem ler ou escrever caches."""
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.ndim != 3 or landmarks.shape[1:] != (self.landmark_count, 2) or len(landmarks) == 0:
            raise ValueError(f"Esperado um array não vazio com formato (frames, {self.landmark_count}, 2).")
        if self.imputation:
            table = pd.DataFrame(landmarks.reshape(len(landmarks), -1), columns=self._columns())
            landmarks = self._impute(table)
        return self.anchor(landmarks)
