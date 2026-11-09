"""Viola-Jones face detection (Haar features + AdaBoost cascade, OpenCV pre-trained model).

Used for two things:
  1. remove the face from the skin mask, so it is never mistaken for a hand;
  2. initialize the head tracker used for the dodge.
"""

from __future__ import annotations

import cv2
import numpy as np

from kombat_vision.config import FaceConfig

Box = tuple[int, int, int, int]  # x, y, w, h


class FaceDetector:
    def __init__(self, cfg: FaceConfig):
        self.cfg = cfg
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")
        self._frame_count = 0

    def should_run(self) -> bool:
        """Viola-Jones is the slowest step, so it runs every `detect_every` frames."""
        self._frame_count += 1
        return self._frame_count % max(1, self.cfg.detect_every) == 0

    def detect(self, gray: np.ndarray) -> Box | None:
        """Return the largest frontal face, or None."""
        equalized = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            equalized,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=self.cfg.min_neighbors,
            minSize=(self.cfg.min_size, self.cfg.min_size),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        return int(x), int(y), int(w), int(h)


def padded_box(box: Box, padding: float, shape: tuple[int, ...]) -> Box:
    """Enlarge a box by `padding` (ratio) on each side, extra margin below for the neck."""
    x, y, w, h = box
    px, py = int(w * padding), int(h * padding)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1 = min(shape[1], x + w + px)
    y1 = min(shape[0], y + h + int(py * 2.2))
    return x0, y0, x1 - x0, y1 - y0
