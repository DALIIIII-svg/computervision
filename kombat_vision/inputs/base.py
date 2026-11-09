"""Interface shared by every input source (keyboard, CV pipeline, MediaPipe baseline)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from kombat_vision.actions import Actions


class InputSource(ABC):
    name = "input"

    def start(self) -> None:
        """Open devices / start threads."""

    @abstractmethod
    def poll(self) -> Actions:
        """Actions since the last poll (events fire once, held states are current)."""

    @property
    def ready(self) -> bool:
        """False while the source is still setting up (background learning, calibration)."""
        return True

    @property
    def message(self) -> str:
        return ""

    @property
    def progress(self) -> float | None:
        return None

    def status_lines(self) -> list[str]:
        return []

    def on_key(self, key: int) -> None:
        """Keys forwarded from the game window (e.g. recalibrate)."""

    def thumbnail(self) -> np.ndarray | None:
        return None

    def show_debug(self) -> None:
        """Show a debug window (called from the main thread)."""

    def close(self) -> None:
        """Release devices / stop threads."""
