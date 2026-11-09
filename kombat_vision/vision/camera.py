"""Threaded webcam capture (also plays a video file, which is handy for repeatable tests).

Reading the camera in its own thread means the pipeline always works on the
newest frame instead of waiting for the driver (lower latency).
"""

from __future__ import annotations

import platform
import threading
import time

import cv2
import numpy as np

from kombat_vision.config import CameraConfig


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "Camera":
        self._is_file = isinstance(self.cfg.index, str)
        if self._is_file:
            self._cap = cv2.VideoCapture(self.cfg.index)
        else:
            windows = self.cfg.use_directshow and platform.system() == "Windows"
            self._cap = cv2.VideoCapture(self.cfg.index, cv2.CAP_DSHOW if windows else cv2.CAP_ANY)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video {self.cfg.index!r}. Try another index in the config.")
        if self._is_file:
            self._delay = 1.0 / (self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                if self._is_file:      # loop the video
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            if self._is_file:
                time.sleep(self._delay)
            if self.cfg.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self) -> tuple[int, np.ndarray | None]:
        """Return (frame_id, frame). frame_id lets callers skip frames they already processed."""
        with self._lock:
            if self._frame is None:
                return 0, None
            return self._frame_id, self._frame.copy()

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
