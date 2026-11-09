"""Base class for camera-driven inputs.

A worker thread reads frames and runs a processor; the game thread only collects
the latest actions. This keeps the game at 60 FPS even if vision runs at 20-30 FPS.
"""

from __future__ import annotations

import threading
import time
from abc import abstractmethod

import cv2
import numpy as np

from kombat_vision.actions import Actions, EVENT_ACTIONS
from kombat_vision.config import Config
from kombat_vision.inputs.base import InputSource
from kombat_vision.vision.camera import Camera


class CameraInput(InputSource):
    window_name = "Kombat Vision - debug"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.camera = Camera(cfg.camera)
        self._lock = threading.Lock()
        self._pending = Actions()
        self._held = Actions()
        self._result = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._commands: list[int] = []
        self.error: str | None = None
        self._window_open = False

    # ------------------------------------------------------ to implement
    @abstractmethod
    def process(self, frame: np.ndarray):
        """Run on the worker thread. Return a result object with an `actions` attribute."""

    @abstractmethod
    def debug_image(self, result) -> np.ndarray | None:
        """Build the debug image for a result (main thread)."""

    def handle_command(self, key: int) -> None:
        """Called on the worker thread for keys forwarded by the game."""

    # ----------------------------------------------------------- thread
    def start(self) -> None:
        self.camera.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-worker", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        last_id = -1
        while self._running:
            frame_id, frame = self.camera.read()
            if frame is None or frame_id == last_id:
                time.sleep(0.002)
                continue
            last_id = frame_id
            with self._lock:
                commands, self._commands = self._commands, []
            for key in commands:
                self.handle_command(key)
            try:
                result = self.process(frame)
            except Exception as exc:  # keep the game alive, show the error
                self.error = f"{type(exc).__name__}: {exc}"
                continue
            with self._lock:
                self._result = result
                for name in EVENT_ACTIONS:
                    if getattr(result.actions, name):
                        setattr(self._pending, name, True)
                self._held = result.actions

    # ---------------------------------------------------------- game side
    def poll(self) -> Actions:
        with self._lock:
            actions = self._pending
            self._pending = Actions()
            actions.block = self._held.block
            actions.move_x = self._held.move_x
        return actions

    def latest(self):
        with self._lock:
            return self._result

    def on_key(self, key: int) -> None:
        with self._lock:
            self._commands.append(key)

    def show_debug(self) -> None:
        result = self.latest()
        if result is None:
            return
        img = self.debug_image(result)
        if img is not None:
            cv2.imshow(self.window_name, img)
            cv2.waitKey(1)
            self._window_open = True

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.camera.release()
        if self._window_open:
            cv2.destroyAllWindows()
