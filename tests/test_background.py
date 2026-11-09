import cv2
import numpy as np
import pytest

from kombat_vision.config import BackgroundConfig
from kombat_vision.vision.background import create_background_model
from tests.synthetic import background


@pytest.mark.parametrize("method", ["mog2", "codebook"])
def test_detects_new_object(method):
    cfg = BackgroundConfig(method=method, warmup_frames=30)
    model = create_background_model(cfg)
    for i in range(30):
        model.apply(cv2.resize(background(seed=i), (320, 240)))
    assert model.ready
    frame = cv2.resize(background(seed=99), (320, 240))
    cv2.rectangle(frame, (120, 60), (200, 200), (30, 200, 240), -1)
    mask = model.apply(frame)
    inside = mask[70:190, 130:190].mean() / 255
    outside = mask[:, :100].mean() / 255
    assert inside > 0.9, f"object not detected ({inside:.2f})"
    assert outside < 0.05, f"too much false foreground ({outside:.2f})"
