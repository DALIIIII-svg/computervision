"""Skin color model in YCrCb, with per-user calibration.

In YCrCb, skin of all tones occupies a compact region of the (Cr, Cb) plane,
while Y (brightness) varies a lot with lighting. So we threshold Cr and Cb and
mostly ignore Y. Calibration samples the user's own hand to tighten the ranges.
"""

from __future__ import annotations

import cv2
import numpy as np

from kombat_vision.config import SkinConfig


class SkinModel:
    def __init__(self, cfg: SkinConfig):
        self.cfg = cfg
        self.cr_range = tuple(cfg.default_cr)
        self.cb_range = tuple(cfg.default_cb)
        self.calibrated = False
        self._samples: list[np.ndarray] = []
        k = cfg.morph_kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # ------------------------------------------------------------ calibration
    def roi_box(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Calibration box in pixels (x, y, w, h)."""
        h, w = shape[:2]
        rx, ry, rw, rh = self.cfg.roi
        return int(rx * w), int(ry * h), int(rw * w), int(rh * h)

    def add_calibration_sample(self, ycrcb: np.ndarray) -> float:
        """Collect skin pixels from the calibration box. Returns progress in [0, 1]."""
        x, y, w, h = self.roi_box(ycrcb.shape)
        patch = ycrcb[y:y + h, x:x + w].reshape(-1, 3)
        patch = patch[patch[:, 0] >= self.cfg.min_y]
        if patch.size:
            self._samples.append(patch)
        progress = len(self._samples) / max(1, self.cfg.calibration_frames)
        if progress >= 1.0:
            self._finish_calibration()
        return min(1.0, progress)

    def _finish_calibration(self) -> None:
        pixels = np.concatenate(self._samples, axis=0)
        # Percentiles instead of min/max: robust to a few background pixels in the box.
        cr_lo, cr_hi = np.percentile(pixels[:, 1], [5, 95])
        cb_lo, cb_hi = np.percentile(pixels[:, 2], [5, 95])
        self.cr_range = (int(max(0, cr_lo - self.cfg.margin_cr)), int(min(255, cr_hi + self.cfg.margin_cr)))
        self.cb_range = (int(max(0, cb_lo - self.cfg.margin_cb)), int(min(255, cb_hi + self.cfg.margin_cb)))
        self.calibrated = True
        self._samples.clear()

    def reset_calibration(self) -> None:
        self.cr_range = tuple(self.cfg.default_cr)
        self.cb_range = tuple(self.cfg.default_cb)
        self.calibrated = False
        self._samples.clear()

    # ----------------------------------------------------------- segmentation
    def segment(self, ycrcb: np.ndarray) -> np.ndarray:
        lower = np.array([self.cfg.min_y, self.cr_range[0], self.cb_range[0]], np.uint8)
        upper = np.array([255, self.cr_range[1], self.cb_range[1]], np.uint8)
        return cv2.inRange(ycrcb, lower, upper)

    def clean(self, mask: np.ndarray) -> np.ndarray:
        """Opening removes small noise, closing fills small holes."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
