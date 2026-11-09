"""Hand extraction: from the skin mask to one clean hand contour.

Steps:
  1. candidate = skin AND foreground AND NOT face
  2. connected components -> pick the blob closest to the tracked hand (or the largest)
  3. region growing from the palm center (gradual Cr/Cb similarity) to refine the blob
  4. Canny edges cut the mask (separates fingers that touch each other)
  5. wrist cut using the palm circle (distance transform), so the forearm does not count
  6. outer contour + palm center/radius for the finger counter
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kombat_vision.config import HandConfig

Box = tuple[int, int, int, int]


@dataclass
class HandObservation:
    contour: np.ndarray                  # N x 1 x 2 int32
    mask: np.ndarray                     # full-frame uint8 mask of the hand only
    box: Box                             # bounding box
    palm_center: tuple[int, int]
    palm_radius: float
    area: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2.0, y + h / 2.0


class HandExtractor:
    def __init__(self, cfg: HandConfig):
        self.cfg = cfg
        self._kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def extract(self, skin: np.ndarray, foreground: np.ndarray, face_box: Box | None,
                ycrcb: np.ndarray, gray: np.ndarray,
                predicted_center: tuple[float, float] | None = None,
                ) -> tuple[HandObservation | None, np.ndarray]:
        """Return (hand or None, candidate mask used for debugging)."""
        h, w = skin.shape
        min_area = self.cfg.min_area_ratio * h * w

        # 1. Candidate pixels.
        fg = cv2.dilate(foreground, self._kernel7)  # be lenient with the foreground mask
        candidate = cv2.bitwise_and(skin, fg)
        if face_box is not None:
            x, y, bw, bh = face_box
            candidate[y:y + bh, x:x + bw] = 0

        # 2. Connected components.
        component = self._pick_component(candidate, min_area, predicted_center)
        if component is None:
            return None, candidate

        hand = component
        # 3. Region growing.
        if self.cfg.use_region_growing:
            grown = self._region_grow(ycrcb, component, allowed=fg, face_box=face_box)
            if grown is not None:
                hand = grown
        # 4. Canny split.
        if self.cfg.use_canny_split:
            hand = self._canny_split(hand, gray)
        # 5. Wrist cut.
        if self.cfg.wrist_cut:
            hand = self._wrist_cut(hand)

        hand = self._largest_component(hand)
        if hand is None or cv2.countNonZero(hand) < min_area:
            return None, candidate
        return self._describe(hand), candidate

    # ----------------------------------------------------------------- helpers
    def _pick_component(self, mask: np.ndarray, min_area: float,
                        predicted: tuple[float, float] | None) -> np.ndarray | None:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best, best_score = None, None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            if predicted is not None:
                dist = np.hypot(*(centroids[i] - np.array(predicted)))
                score = -dist + 0.002 * area   # closest to the tracked hand, size as tie-breaker
            else:
                score = area
            if best_score is None or score > best_score:
                best, best_score = i, score
        if best is None:
            return None
        return np.where(labels == best, 255, 0).astype(np.uint8)

    @staticmethod
    def _palm(mask: np.ndarray) -> tuple[tuple[int, int], float]:
        """Palm = largest inscribed circle = maximum of the distance transform."""
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, radius, _, center = cv2.minMaxLoc(dist)
        return (int(center[0]), int(center[1])), float(radius)

    def _region_grow(self, ycrcb: np.ndarray, component: np.ndarray, allowed: np.ndarray,
                     face_box: Box | None) -> np.ndarray | None:
        """Grow from the palm center: a pixel joins if its (Cr, Cb) is close to the seed's.

        Comparing with the seed (fixed range) rather than with the neighbor (gradual)
        stops the growth at the blurred hand border instead of creeping through it.
        """
        seed, radius = self._palm(component)
        if radius < 2:
            return None
        chroma = ycrcb.copy()
        chroma[..., 0] = 0                     # ignore brightness, compare only Cr and Cb
        h, w = component.shape
        blocked = np.zeros((h + 2, w + 2), np.uint8)
        forbidden = cv2.bitwise_not(allowed)
        if face_box is not None:
            x, y, bw, bh = face_box
            forbidden[y:y + bh, x:x + bw] = 255
        blocked[1:-1, 1:-1] = (forbidden > 0).astype(np.uint8)
        tol = self.cfg.region_grow_tolerance
        flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
        cv2.floodFill(chroma, blocked, seed, (0, 0, 0), (0, tol, tol), (0, tol, tol), flags)
        grown = np.where(blocked[1:-1, 1:-1] == 255, 255, 0).astype(np.uint8)
        # Guard against leaking into a large similar-colored area.
        if cv2.countNonZero(grown) > 2.5 * cv2.countNonZero(component):
            return None
        return cv2.morphologyEx(grown, cv2.MORPH_CLOSE, self._kernel3)

    def _canny_split(self, mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            return mask
        edges = cv2.Canny(gray[y:y + h, x:x + w], self.cfg.canny_low, self.cfg.canny_high)
        out = mask.copy()
        roi = out[y:y + h, x:x + w]
        # Only edges well inside the mask: the outline itself is already the mask border,
        # cutting along it would just leave thin slivers.
        interior = cv2.erode(roi, self._kernel7)
        roi[(edges > 0) & (interior > 0)] = 0
        return cv2.morphologyEx(out, cv2.MORPH_OPEN, self._kernel3)

    def _wrist_cut(self, mask: np.ndarray) -> np.ndarray:
        """Remove the forearm: everything well below the palm circle (hand assumed pointing up)."""
        if cv2.countNonZero(mask) == 0:
            return mask
        (cx, cy), radius = self._palm(mask)
        cut_y = int(cy + self.cfg.wrist_cut_factor * radius)
        out = mask.copy()
        out[max(0, cut_y):, :] = 0
        return out

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray | None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return None
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == best, 255, 0).astype(np.uint8)

    def _describe(self, mask: np.ndarray) -> HandObservation:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        center, radius = self._palm(mask)
        return HandObservation(
            contour=contour,
            mask=mask,
            box=tuple(int(v) for v in cv2.boundingRect(contour)),
            palm_center=center,
            palm_radius=radius,
            area=float(cv2.contourArea(contour)),
        )
