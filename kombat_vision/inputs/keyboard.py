"""Keyboard input: for playing without a camera and for debugging the game."""

from __future__ import annotations

import pygame

from kombat_vision.actions import Actions
from kombat_vision.inputs.base import InputSource

KEY_HELP = "keys: A/D move  J punch  K kick  L block  U special  S dodge  R restart  ESC quit"

EVENT_KEYS = {
    pygame.K_j: "punch",
    pygame.K_k: "kick",
    pygame.K_u: "special",
    pygame.K_s: "dodge",
}


class KeyboardInput(InputSource):
    name = "keyboard"

    def __init__(self):
        self._pending = Actions()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key in EVENT_KEYS:
            setattr(self._pending, EVENT_KEYS[event.key], True)

    def poll(self) -> Actions:
        actions, self._pending = self._pending, Actions()
        keys = pygame.key.get_pressed()
        actions.block = bool(keys[pygame.K_l])
        actions.move_x = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
        return actions
