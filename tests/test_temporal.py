from kombat_vision.actions import Gesture
from kombat_vision.config import GesturesConfig
from kombat_vision.vision.temporal import GestureActionMapper, GestureStabilizer


def test_stabilizer_ignores_short_glitches():
    s = GestureStabilizer(stable_frames=3)
    seq = [Gesture.FIST] * 4 + [Gesture.FOUR] + [Gesture.FIST] * 3
    actives = [s.update(g)[0] for g in seq]
    assert Gesture.FOUR not in actives


def test_attack_fires_once_on_gesture_start():
    m = GestureActionMapper(GesturesConfig(stable_frames=2, hold_repeat_s=0))
    punches = [m.update(Gesture.ONE, 0.5)[0].punch for _ in range(10)]
    assert sum(punches) == 1


def test_block_is_held_and_move_zones():
    m = GestureActionMapper(GesturesConfig(stable_frames=1))
    actions, _ = m.update(Gesture.PALM, 0.1)
    assert actions.block and actions.move_x == -1
    actions, _ = m.update(Gesture.PALM, 0.9)
    assert actions.block and actions.move_x == 1
