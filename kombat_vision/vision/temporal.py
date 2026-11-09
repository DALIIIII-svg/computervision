"""Turns a noisy per-frame gesture into clean game actions.

- Stabilizer: a gesture becomes "active" only after `stable_frames` identical frames
  (transition shapes between two gestures are ignored).
- Mapper: active gesture -> game action (from the config), fired once when the
  gesture starts, optionally repeated while it is held.
"""

from __future__ import annotations

import time

from kombat_vision.actions import Actions, Gesture, ALL_ACTIONS, HELD_ACTIONS
from kombat_vision.config import GesturesConfig


class GestureStabilizer:
    def __init__(self, stable_frames: int):
        self.stable_frames = stable_frames
        self._candidate = Gesture.NONE
        self._count = 0
        self.active = Gesture.NONE

    def update(self, gesture: Gesture) -> tuple[Gesture, bool]:
        """Return (active gesture, changed?)."""
        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = gesture, 1
        if self._count >= self.stable_frames and self._candidate != self.active:
            self.active = self._candidate
            return self.active, True
        return self.active, False


class GestureActionMapper:
    def __init__(self, cfg: GesturesConfig):
        self.cfg = cfg
        self.stabilizer = GestureStabilizer(cfg.stable_frames)
        mapping = {}
        for gesture_name, action in cfg.mapping.items():
            if action not in ALL_ACTIONS and action != "none":
                raise ValueError(f"Unknown action '{action}' for gesture '{gesture_name}'")
            mapping[Gesture(gesture_name)] = action
        self.mapping = mapping
        self._last_fire = 0.0

    def update(self, gesture: Gesture, hand_x_ratio: float | None) -> tuple[Actions, Gesture]:
        active, changed = self.stabilizer.update(gesture)
        actions = Actions()
        action = self.mapping.get(active, "none")
        now = time.monotonic()

        if action in HELD_ACTIONS:
            setattr(actions, action, True)
        elif action != "none":
            repeat = self.cfg.hold_repeat_s > 0 and now - self._last_fire >= self.cfg.hold_repeat_s
            if changed or repeat:
                setattr(actions, action, True)
                self._last_fire = now

        if hand_x_ratio is not None and active != Gesture.NONE:
            left, right = self.cfg.move_zones
            if hand_x_ratio < left:
                actions.move_x = -1
            elif hand_x_ratio > right:
                actions.move_x = 1
        return actions, active
