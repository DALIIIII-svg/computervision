"""Common interface for background subtraction methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BackgroundModel(ABC):
    """Learns the empty scene during `warmup_frames`, then returns a foreground mask (255 = player)."""

    def __init__(self, warmup_frames: int):
        self.warmup_frames = warmup_frames
        self.frames_seen = 0

    @property
    def ready(self) -> bool:
        return self.frames_seen >= self.warmup_frames

    @property
    def progress(self) -> float:
        return min(1.0, self.frames_seen / max(1, self.warmup_frames))

    @abstractmethod
    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Update the model (if still learning) and return a uint8 mask."""

    def reset(self) -> None:
        self.frames_seen = 0


class NoBackground(BackgroundModel):
    """Disables background subtraction (everything is foreground)."""

    def __init__(self):
        super().__init__(warmup_frames=0)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        return np.full(frame_bgr.shape[:2], 255, np.uint8)
