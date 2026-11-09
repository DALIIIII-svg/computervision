"""Mixture of Gaussians background subtraction (OpenCV MOG2, Zivkovic 2004).

Each pixel is modelled by a mixture of Gaussians over its color history.
A pixel that does not fit the background Gaussians is foreground.
"""

from __future__ import annotations

import cv2
import numpy as np

from kombat_vision.config import Mog2Config
from kombat_vision.vision.background.base import BackgroundModel


class Mog2Background(BackgroundModel):
    def __init__(self, cfg: Mog2Config, warmup_frames: int):
        super().__init__(warmup_frames)
        self.cfg = cfg
        self._create()

    def _create(self) -> None:
        self._model = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history,
            varThreshold=self.cfg.var_threshold,
            detectShadows=self.cfg.detect_shadows,
        )

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Fast learning during warm-up, then almost frozen: otherwise a player
        # who holds still would slowly fade into the background.
        rate = self.cfg.learning_rate if not self.ready else self.cfg.learning_rate_after
        mask = self._model.apply(frame_bgr, learningRate=rate)
        self.frames_seen += 1
        # MOG2 marks shadows as 127: drop them, keep only real foreground (255).
        return np.where(mask == 255, 255, 0).astype(np.uint8)

    def reset(self) -> None:
        super().reset()
        self._create()
