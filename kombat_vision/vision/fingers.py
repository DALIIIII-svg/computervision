"""Finger counting with the convex hull and convexity defects (no training data needed).

The convex hull is the "rubber band" around the hand. Convexity defects are the
places where the contour goes deep inside that band: the valleys between fingers.
A valley counts if:
  - it is deep enough compared with the palm radius (not just contour noise),
  - the angle between the two neighboring fingertips is sharp (< 90 deg),
  - it is not below the palm center (valleys between fingers are above it).
Then: fingers = valleys + 1.

With zero valleys we cannot tell 0 from 1 by valleys alone, so we look at how far
the farthest contour point is from the palm: a fist is compact, a raised thumb or
finger sticks out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from kombat_vision.actions import Gesture
from kombat_vision.config import FingersConfig
from kombat_vision.vision.hand import HandObservation


@dataclass
class FingerResult:
    gesture: Gesture
    count: int
    defects: list = field(default_factory=list)    # [(start, end, far)] valid valleys
    hull: np.ndarray | None = None
    tip: tuple[int, int] | None = None              # farthest point (for 0/1 decision)


class FingerCounter:
    def __init__(self, cfg: FingersConfig):
        self.cfg = cfg

    def classify(self, hand: HandObservation | None) -> FingerResult:
        if hand is None or len(hand.contour) < 5:
            return FingerResult(Gesture.NONE, -1)

        contour = cv2.approxPolyDP(hand.contour, 0.005 * cv2.arcLength(hand.contour, True), True)
        if len(contour) < 4:
            return FingerResult(Gesture.NONE, -1)
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        hull_pts = cv2.convexHull(contour)

        valleys = []
        try:
            defects = cv2.convexityDefects(contour, hull_idx) if len(hull_idx) > 3 else None
        except cv2.error:  # self-intersecting contours can make OpenCV refuse
            defects = None
        cx, cy = hand.palm_center
        r = max(hand.palm_radius, 1.0)

        if defects is not None:
            for s, e, f, d in defects[:, 0]:
                start, end, far = (tuple(int(v) for v in contour[i][0]) for i in (s, e, f))
                depth = d / 256.0
                if depth < self.cfg.defect_min_depth * r:
                    continue
                if _angle(start, far, end) > self.cfg.defect_max_angle:
                    continue
                if far[1] > cy + 0.5 * r:      # valley below the palm: wrist/arm artifact
                    continue
                valleys.append((start, end, far))

        if valleys:
            count = min(len(valleys) + 1, 5)
            return FingerResult(Gesture.from_count(count), count, valleys, hull_pts)

        # No valleys: fist or one raised finger/thumb?
        pts = contour[:, 0, :]
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        i = int(np.argmax(dists))
        tip = (int(pts[i, 0]), int(pts[i, 1]))
        pointing_up = tip[1] < cy - 0.5 * r
        if dists[i] / r >= self.cfg.extended_ratio and pointing_up:
            return FingerResult(Gesture.ONE, 1, [], hull_pts, tip)
        return FingerResult(Gesture.FIST, 0, [], hull_pts, tip)


def _angle(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> float:
    """Angle at b (degrees) between segments b->a and b->c."""
    ba = np.subtract(a, b).astype(float)
    bc = np.subtract(c, b).astype(float)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
