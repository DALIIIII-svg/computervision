"""Per-stage latency measurement, used for the debug view and the report."""

from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    """Measures how long each pipeline stage takes, smoothed with an exponential moving average.

        timer = StageTimer()
        with timer("skin"):
            ...
        timer.ms  # {"skin": 1.8, ...}
    """

    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.ms: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            previous = self.ms.get(name)
            self.ms[name] = elapsed if previous is None else (
                self.smoothing * previous + (1 - self.smoothing) * elapsed)

    def total(self) -> float:
        return sum(v for k, v in self.ms.items() if k != "total")


class FpsCounter:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else self.smoothing * self.fps + (1 - self.smoothing) * inst
        self._last = now
        return self.fps
