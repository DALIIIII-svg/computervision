"""Synthetic images for tests: a simple drawn 'hand' (palm + fingers + wrist) on a background."""

from __future__ import annotations

import math

import cv2
import numpy as np

SKIN_BGR = (100, 140, 205)


def background(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (150, 110, 60)                               # bluish wall, clearly not skin
    cv2.rectangle(img, (40, 300), (200, 470), (60, 120, 60), -1)   # a green "plant"
    noise = rng.integers(-6, 7, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def draw_hand(img: np.ndarray, center: tuple[int, int], fingers: int, palm_r: int = 56,
              color=SKIN_BGR) -> np.ndarray:
    out = img.copy()
    cx, cy = center
    # wrist / forearm going down
    cv2.rectangle(out, (cx - int(palm_r * 0.75), cy), (cx + int(palm_r * 0.75), cy + 4 * palm_r), color, -1)
    cv2.circle(out, (cx, cy), palm_r, color, -1)
    if fingers > 0:
        spread = 14 * (fingers - 1)          # fingers about 28 deg apart, like a real hand
        angles = [0.0] if fingers == 1 else np.linspace(-spread, spread, fingers)
        length, thick = 2.3 * palm_r, int(0.36 * palm_r)
        for a in angles:
            rad = math.radians(a)
            x0 = int(cx + 0.6 * palm_r * math.sin(rad))
            y0 = int(cy - 0.6 * palm_r * math.cos(rad))
            x1 = int(cx + length * math.sin(rad))
            y1 = int(cy - length * math.cos(rad))
            cv2.line(out, (x0, y0), (x1, y1), color, thick)
            cv2.circle(out, (x1, y1), thick // 2, color, -1)
    return out


def hand_mask(fingers: int, size=(240, 320), center=(160, 140), palm_r=28) -> np.ndarray:
    canvas = np.zeros((*size, 3), np.uint8)
    drawn = draw_hand(canvas, center, fingers, palm_r, color=(255, 255, 255))
    return cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY)
