"""Dodge detection from the tracked head.

The head is tracked (Viola-Jones + CamShift + Kalman). A dodge is detected when:
  - the head moves down fast (Kalman vertical velocity), or
  - the head is well below its usual (rest) position.
Speeds and distances are divided by the face height, so it works at any distance.

Note: the frontal Viola-Jones cascade often loses the face when you bow your head.
That is fine: CamShift keeps tracking, and the Kalman velocity stays informative.
"""

from __future__ import annotations

import time

from kombat_vision.config import DodgeConfig
from kombat_vision.vision.tracking import TrackState


class DodgeDetector:
    def __init__(self, cfg: DodgeConfig):
        self.cfg = cfg
        self.rest_y: float | None = None
        self._last_dodge = 0.0
        self.last_score = 0.0

    def update(self, head: TrackState, face_height: float | None) -> bool:
        if not head.active or not face_height:
            return False
        _, cy = head.center
        _, vy = head.velocity
        speed = vy / face_height                       # face heights per frame (down > 0)

        if self.rest_y is None:
            self.rest_y = cy
        drop = (cy - self.rest_y) / face_height

        # Slowly follow the rest position when the head is calm.
        if abs(speed) < 0.02 and drop < 0.3:
            self.rest_y = 0.97 * self.rest_y + 0.03 * cy

        self.last_score = max(speed / self.cfg.velocity_threshold, drop / self.cfg.drop_threshold)
        now = time.monotonic()
        if self.last_score >= 1.0 and now - self._last_dodge > self.cfg.cooldown_s:
            self._last_dodge = now
            return True
        return False

    def reset(self) -> None:
        self.rest_y = None
