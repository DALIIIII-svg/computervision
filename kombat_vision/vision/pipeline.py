"""The perception pipeline: camera frame -> game Actions.

    frame
      -> preprocess (resize, blur, YCrCb)
      -> background subtraction (MOG2 or Codebook)         -> foreground mask
      -> Viola-Jones face + CamShift/Kalman head tracking  -> face box, head motion
      -> skin segmentation (calibrated YCrCb)              -> skin mask
      -> hand extraction (components, region growing, Canny, wrist cut)
      -> hand tracking (CamShift + Kalman)                 -> hand position
      -> finger counting (convex hull + convexity defects) -> gesture
      -> dodge detection (head velocity / drop)
      -> temporal stabilizer + gesture->action mapping     -> Actions

Startup phases:
    BACKGROUND  step out of the frame while the empty scene is learned
    CALIBRATE   hold your open hand in the box to learn your skin color
    RUNNING     play
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from kombat_vision.actions import Actions, Gesture
from kombat_vision.config import Config
from kombat_vision.utils.timing import FpsCounter, StageTimer
from kombat_vision.vision.background import NoBackground, create_background_model
from kombat_vision.vision.dodge import DodgeDetector
from kombat_vision.vision.face import FaceDetector, padded_box
from kombat_vision.vision.fingers import FingerCounter, FingerResult
from kombat_vision.vision.hand import HandExtractor, HandObservation
from kombat_vision.vision.preprocessing import Preprocessor
from kombat_vision.vision.skin import SkinModel
from kombat_vision.vision.temporal import GestureActionMapper
from kombat_vision.vision.tracking import TrackState, Tracker, back_project, hue_histogram


class Phase(str, Enum):
    BACKGROUND = "background"
    CALIBRATE = "calibrate"
    RUNNING = "running"


@dataclass
class PipelineResult:
    phase: Phase
    message: str
    progress: float = 1.0
    actions: Actions = field(default_factory=Actions)
    raw_gesture: Gesture = Gesture.NONE
    gesture: Gesture = Gesture.NONE          # stabilized gesture
    fingers: FingerResult | None = None
    hand: HandObservation | None = None
    hand_track: TrackState | None = None
    head_track: TrackState | None = None
    face_box: tuple | None = None
    dodge_score: float = 0.0
    frame: np.ndarray | None = None          # processed BGR frame
    foreground: np.ndarray | None = None
    skin: np.ndarray | None = None
    candidate: np.ndarray | None = None
    timings: dict = field(default_factory=dict)
    fps: float = 0.0


class VisionPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.timer = StageTimer()
        self.fps = FpsCounter()
        self.preprocess = Preprocessor(cfg.preprocessing)
        self.background = create_background_model(cfg.background)
        self.skin = SkinModel(cfg.skin)
        self.face = FaceDetector(cfg.face) if cfg.face.enabled else None
        self.hand = HandExtractor(cfg.hand)
        self.fingers = FingerCounter(cfg.fingers)
        self.hand_tracker = Tracker(cfg.tracking)
        self.head_tracker = Tracker(cfg.tracking)
        self.dodge = DodgeDetector(cfg.dodge)
        self.mapper = GestureActionMapper(cfg.gestures)
        self._face_hist = None
        self._face_height: float | None = None
        self.phase = Phase.RUNNING if isinstance(self.background, NoBackground) else Phase.BACKGROUND
        if self.phase == Phase.RUNNING:
            self.phase = Phase.CALIBRATE

    # ------------------------------------------------------------ user controls
    def restart_background(self) -> None:
        if not isinstance(self.background, NoBackground):
            self.background.reset()
            self.phase = Phase.BACKGROUND

    def restart_calibration(self) -> None:
        self.skin.reset_calibration()
        self.phase = Phase.CALIBRATE

    def skip_calibration(self) -> None:
        """Use the default skin ranges."""
        if self.phase == Phase.CALIBRATE:
            self.phase = Phase.RUNNING

    # ------------------------------------------------------------------- main
    def process(self, frame: np.ndarray) -> PipelineResult:
        t = self.timer
        with t("total"):
            with t("preprocess"):
                bgr, ycrcb, gray = self.preprocess(frame)

            with t("background"):
                foreground = self.background.apply(bgr)

            if self.phase == Phase.BACKGROUND:
                if self.background.ready:
                    self.phase = Phase.CALIBRATE
                return self._result(Phase.BACKGROUND, "Step out of the frame: learning the background",
                                    self.background.progress, bgr, foreground)

            with t("face"):
                face_box = self._update_head(bgr, gray)
                masked_face = padded_box(face_box, self.cfg.face.mask_padding, bgr.shape) if face_box else None

            if self.phase == Phase.CALIBRATE:
                progress = self.skin.add_calibration_sample(ycrcb)
                if self.skin.calibrated:
                    self.phase = Phase.RUNNING
                res = self._result(Phase.CALIBRATE, "Hold your open hand inside the box",
                                   progress, bgr, foreground)
                res.face_box = face_box
                return res

            with t("skin"):
                skin = self.skin.clean(self.skin.segment(ycrcb))

            with t("hand"):
                predicted = self.hand_tracker.kf.position if self.hand_tracker.kf.initialized else None
                hand, candidate = self.hand.extract(skin, foreground, masked_face, ycrcb, gray, predicted)

            with t("tracking"):
                hand_state = self.hand_tracker.update(hand.box if hand else None, candidate)

            with t("fingers"):
                fingers = self.fingers.classify(hand)

            with t("dodge"):
                dodge = self.dodge.update(self.head_tracker.state, self._face_height)

            with t("mapping"):
                x_ratio = hand_state.center[0] / bgr.shape[1] if hand_state.active else None
                actions, stable = self.mapper.update(fingers.gesture, x_ratio)
                actions.dodge = actions.dodge or dodge

        res = self._result(Phase.RUNNING, "", 1.0, bgr, foreground)
        res.actions, res.raw_gesture, res.gesture = actions, fingers.gesture, stable
        res.fingers, res.hand, res.hand_track = fingers, hand, hand_state
        res.face_box, res.skin, res.candidate = face_box, skin, candidate
        res.dodge_score = self.dodge.last_score
        return res

    # ---------------------------------------------------------------- helpers
    def _update_head(self, bgr: np.ndarray, gray: np.ndarray):
        if self.face is None:
            return None
        detection = self.face.detect(gray) if self.face.should_run() else None
        if detection is not None:
            self._face_height = float(detection[3])
            if self._face_hist is None or not self.head_tracker.state.active:
                self._face_hist = hue_histogram(bgr, detection)
        prob = back_project(bgr, self._face_hist) if self._face_hist is not None else None
        state = self.head_tracker.update(detection, prob)
        if not state.active:
            self._face_hist = None
            self.dodge.reset()
            return None
        return state.box

    def _result(self, phase: Phase, message: str, progress: float,
                bgr: np.ndarray, foreground: np.ndarray) -> PipelineResult:
        return PipelineResult(
            phase=phase, message=message, progress=progress,
            frame=bgr, foreground=foreground,
            head_track=self.head_tracker.state,
            timings=dict(self.timer.ms), fps=self.fps.tick(),
        )
