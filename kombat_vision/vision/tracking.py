"""Tracking: a constant-velocity Kalman filter + CamShift (Mean Shift with adaptive window).

- Mean Shift / CamShift moves a window to the densest area of a probability image
  (here: the hand mask, or the face's hue back-projection). It only needs a starting
  window, so it can follow the target on frames where the detector is not run or fails.
- The Kalman filter smooths the measured center and estimates velocity. Its prediction
  step also gives a position when the target is briefly lost, and the predicted
  position is used as the search start for the next frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kombat_vision.config import TrackingConfig

Box = tuple[int, int, int, int]


class KalmanFilter2D:
    """State [x, y, vx, vy], measurement [x, y], constant-velocity model (dt = 1 frame).

    Written with NumPy (instead of cv2.KalmanFilter) so every equation is visible:
        predict:  x = F x            P = F P F^T + Q
        update:   K = P H^T (H P H^T + R)^-1
                  x = x + K (z - H x)  P = (I - K H) P
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 8.0):
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], np.float64)
        q = process_noise
        # Discrete white-noise acceleration model.
        self.Q = q * np.array([[0.25, 0, 0.5, 0],
                               [0, 0.25, 0, 0.5],
                               [0.5, 0, 1, 0],
                               [0, 0.5, 0, 1]], np.float64)
        self.R = (measurement_noise ** 2) * np.eye(2)
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1000.0
        self.initialized = False

    def init(self, cx: float, cy: float) -> None:
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([self.R[0, 0], self.R[1, 1], 100.0, 100.0])
        self.initialized = True

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, cx: float, cy: float) -> np.ndarray:
        if not self.initialized:
            self.init(cx, cy)
            return self.x[:2].copy()
        z = np.array([cx, cy])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].copy()

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


@dataclass
class TrackState:
    box: Box | None = None              # current window
    center: tuple[float, float] | None = None   # Kalman-filtered center
    velocity: tuple[float, float] = (0.0, 0.0)  # pixels / frame
    lost_frames: int = 0
    measured: bool = False              # True if this frame had a real measurement

    @property
    def active(self) -> bool:
        return self.center is not None


class Tracker:
    """Detection when available, CamShift otherwise, Kalman filter on top."""

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.kf = KalmanFilter2D(cfg.process_noise, cfg.measurement_noise)
        self.state = TrackState()
        self._criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def reset(self) -> None:
        self.kf = KalmanFilter2D(self.cfg.process_noise, self.cfg.measurement_noise)
        self.state = TrackState()

    def update(self, detection: Box | None, prob: np.ndarray | None = None) -> TrackState:
        s = self.state
        if self.kf.initialized:
            self.kf.predict()

        measurement: Box | None = detection
        if measurement is None and prob is not None and s.box is not None:
            measurement = self._camshift(prob, self._window_at_prediction(s.box, prob.shape))

        if measurement is not None:
            x, y, w, h = measurement
            self.kf.update(x + w / 2.0, y + h / 2.0)
            s.box, s.lost_frames, s.measured = measurement, 0, True
        else:
            s.lost_frames += 1
            s.measured = False

        if s.lost_frames > self.cfg.max_lost_frames or not self.kf.initialized:
            self.reset()
            return self.state

        s.center = self.kf.position
        s.velocity = self.kf.velocity
        return s

    def _window_at_prediction(self, box: Box, shape: tuple[int, ...]) -> Box:
        """Move the last window to the Kalman-predicted center."""
        _, _, w, h = box
        cx, cy = self.kf.position
        x = int(np.clip(cx - w / 2, 0, shape[1] - 1))
        y = int(np.clip(cy - h / 2, 0, shape[0] - 1))
        return x, y, max(1, min(w, shape[1] - x)), max(1, min(h, shape[0] - y))

    def _camshift(self, prob: np.ndarray, window: Box) -> Box | None:
        if prob[window[1]:window[1] + window[3], window[0]:window[0] + window[2]].sum() == 0:
            return None
        _, new_window = cv2.CamShift(prob, window, self._criteria)
        x, y, w, h = (int(v) for v in new_window)
        if w < 4 or h < 4:
            return None
        return x, y, w, h


def hue_histogram(bgr: np.ndarray, box: Box) -> np.ndarray:
    """Hue histogram of a region, used to back-project the face for CamShift."""
    x, y, w, h = box
    hsv = cv2.cvtColor(bgr[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def back_project(bgr: np.ndarray, hist: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
