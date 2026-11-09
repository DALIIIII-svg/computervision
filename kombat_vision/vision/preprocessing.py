"""Step 1: resize, denoise and convert color space."""

from __future__ import annotations

import cv2
import numpy as np

from kombat_vision.config import PreprocessingConfig


class Preprocessor:
    def __init__(self, cfg: PreprocessingConfig):
        self.cfg = cfg

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (bgr, ycrcb, gray) at processing resolution.

        Smaller frames make every later step faster. YCrCb separates brightness (Y)
        from color (Cr, Cb), which makes skin detection less sensitive to lighting.
        """
        if self.cfg.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.cfg.scale, fy=self.cfg.scale,
                               interpolation=cv2.INTER_AREA)
        k = self.cfg.blur_ksize
        blurred = cv2.GaussianBlur(frame, (k, k), 0) if k and k > 1 else frame
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return blurred, ycrcb, gray
