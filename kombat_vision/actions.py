"""The contract between input sources and the game.

Every input source (keyboard, our CV pipeline, the MediaPipe baseline) produces
the same `Actions` object, so they can be swapped and compared fairly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Gesture(str, Enum):
    NONE = "none"        # no hand visible
    FIST = "fist"        # 0 fingers
    ONE = "one"          # thumb up (or one finger)
    TWO = "two"
    THREE = "three"
    FOUR = "four"
    PALM = "palm"        # 5 fingers, open hand

    @classmethod
    def from_count(cls, count: int) -> "Gesture":
        return {0: cls.FIST, 1: cls.ONE, 2: cls.TWO, 3: cls.THREE,
                4: cls.FOUR, 5: cls.PALM}.get(count, cls.PALM if count > 5 else cls.NONE)


# Actions that are one-shot events (fire once) vs held states.
EVENT_ACTIONS = ("punch", "kick", "special", "dodge")
HELD_ACTIONS = ("block",)
ALL_ACTIONS = EVENT_ACTIONS + HELD_ACTIONS


@dataclass
class Actions:
    move_x: int = 0          # -1 left, 0 none, +1 right
    punch: bool = False
    kick: bool = False
    special: bool = False
    dodge: bool = False
    block: bool = False

    def merge(self, other: "Actions") -> "Actions":
        """Combine two sources (e.g. vision + keyboard): any True wins, keyboard move wins if set."""
        return Actions(
            move_x=other.move_x if other.move_x != 0 else self.move_x,
            punch=self.punch or other.punch,
            kick=self.kick or other.kick,
            special=self.special or other.special,
            dodge=self.dodge or other.dodge,
            block=self.block or other.block,
        )

    def any_event(self) -> bool:
        return any(getattr(self, name) for name in EVENT_ACTIONS)
