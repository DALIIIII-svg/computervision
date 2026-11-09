"""Baseline: MediaPipe Hands (deep learning) + the SAME gesture->action logic as ours.

Only the perception changes, so comparing it with VisionInput measures our
classical pipeline fairly. Install with:  pip install -r requirements-baseline.txt

Uses the MediaPipe Tasks API (HandLandmarker), which exists in both the 0.10.x
and the 1.x releases. The model (~8 MB) is downloaded to models/ on first run.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from kombat_vision.actions import Actions, Gesture
from kombat_vision.config import Config
from kombat_vision.inputs.camera_input import CameraInput
from kombat_vision.utils.timing import FpsCounter, StageTimer
from kombat_vision.vision.temporal import GestureActionMapper

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

TIPS = (8, 12, 16, 20)
PIPS = (6, 10, 14, 18)


@dataclass
class BaselineResult:
    actions: Actions = field(default_factory=Actions)
    raw_gesture: Gesture = Gesture.NONE
    gesture: Gesture = Gesture.NONE
    frame: np.ndarray | None = None
    fps: float = 0.0
    ms: float = 0.0


def count_fingers(lm: np.ndarray) -> int:
    """lm: 21 x 2 normalized landmarks (x, y), image y pointing down."""
    fingers = sum(1 for tip, pip in zip(TIPS, PIPS) if lm[tip, 1] < lm[pip, 1])
    # Thumb is extended if its tip is clearly farther from the middle-finger base than its IP joint.
    d_tip = np.linalg.norm(lm[4] - lm[9])
    d_ip = np.linalg.norm(lm[3] - lm[9])
    thumb = d_tip > 1.15 * d_ip
    return fingers + int(thumb)


class MediaPipeInput(CameraInput):
    name = "baseline (MediaPipe Hands)"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed. Run: pip install -r requirements-baseline.txt") from exc
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading the MediaPipe hand model to {MODEL_PATH} ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts = -1
        self.mapper = GestureActionMapper(cfg.gestures)
        self.timer = StageTimer()
        self.fps = FpsCounter()

    def process(self, frame: np.ndarray) -> BaselineResult:
        with self.timer("total"):
            small = cv2.resize(frame, None, fx=self.cfg.preprocessing.scale, fy=self.cfg.preprocessing.scale)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            ts = max(self._last_ts + 1, int((time.monotonic() - self._t0) * 1000))  # must increase
            self._last_ts = ts
            out = self._landmarker.detect_for_video(image, ts)
            gesture, x_ratio = Gesture.NONE, None
            if out.hand_landmarks:
                lm = np.array([[p.x, p.y] for p in out.hand_landmarks[0]])
                gesture = Gesture.from_count(count_fingers(lm))
                x_ratio = float(lm[:, 0].mean())
                self._draw(small, lm)
            actions, stable = self.mapper.update(gesture, x_ratio)
        cv2.putText(small, stable.value.upper(), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 80), 2)
        return BaselineResult(actions, gesture, stable, small, self.fps.tick(), self.timer.ms["total"])

    @staticmethod
    def _draw(img: np.ndarray, lm: np.ndarray) -> None:
        h, w = img.shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (60, 60, 230), 2)
        for p in pts:
            cv2.circle(img, p, 3, (245, 117, 66), -1)

    def close(self) -> None:
        super().close()
        self._landmarker.close()

    def status_lines(self) -> list[str]:
        res = self.latest()
        if res is None:
            return []
        return [f"gesture: {res.gesture.value}   vision: {res.fps:.0f} FPS, {res.ms:.0f} ms"]

    @property
    def ready(self) -> bool:
        return self.latest() is not None

    @property
    def message(self) -> str:
        return self.error or "Opening camera..."

    def thumbnail(self) -> np.ndarray | None:
        res = self.latest()
        return res.frame if res is not None else None

    def debug_image(self, result: BaselineResult) -> np.ndarray | None:
        return result.frame
