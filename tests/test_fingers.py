import numpy as np
import pytest

from kombat_vision.actions import Gesture
from kombat_vision.config import FingersConfig, HandConfig
from kombat_vision.vision.fingers import FingerCounter
from kombat_vision.vision.hand import HandExtractor
from tests.synthetic import hand_mask


@pytest.mark.parametrize("n,expected", [
    (0, Gesture.FIST), (1, Gesture.ONE), (2, Gesture.TWO),
    (3, Gesture.THREE), (4, Gesture.FOUR), (5, Gesture.PALM),
])
def test_finger_count_on_clean_masks(n, expected):
    mask = hand_mask(n)
    extractor = HandExtractor(HandConfig(use_region_growing=False, use_canny_split=False))
    hand = extractor._describe(extractor._wrist_cut(mask))
    result = FingerCounter(FingersConfig()).classify(hand)
    assert result.gesture == expected, f"{n} fingers -> {result.gesture} ({result.count})"


def test_no_hand_gives_none():
    assert FingerCounter(FingersConfig()).classify(None).gesture == Gesture.NONE
