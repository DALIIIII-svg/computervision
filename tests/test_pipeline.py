"""End-to-end: background learning -> calibration -> gesture, on synthetic frames."""

import pytest

from kombat_vision.actions import Gesture
from kombat_vision.config import Config
from kombat_vision.vision.pipeline import Phase, VisionPipeline
from tests.synthetic import background, draw_hand


@pytest.mark.parametrize("method", ["mog2", "codebook"])
@pytest.mark.parametrize("fingers,expected", [(1, Gesture.ONE), (2, Gesture.TWO),
                                              (4, Gesture.FOUR), (5, Gesture.PALM)])
def test_pipeline_recognizes_gesture(method, fingers, expected):
    cfg = Config()
    cfg.face.enabled = False                 # no face in synthetic frames
    cfg.background.method = method
    cfg.background.warmup_frames = 30
    cfg.skin.calibration_frames = 10
    p = VisionPipeline(cfg)

    for i in range(30):
        p.process(background(seed=i))
    assert p.phase == Phase.CALIBRATE

    for i in range(10):                       # palm inside the calibration box
        p.process(draw_hand(background(seed=100 + i), (320, 200), 0, palm_r=70))
    assert p.phase == Phase.RUNNING

    res = None
    for i in range(8):
        res = p.process(draw_hand(background(seed=200 + i), (320, 260), fingers))
    assert res.gesture == expected, f"raw={res.raw_gesture} stable={res.gesture}"
