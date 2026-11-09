"""All drawing. Fighters are drawn procedurally (no copyrighted sprites).

To use your own sprite sheets later, replace `draw_fighter` and keep the rest.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from kombat_vision.game.fighter import (ATTACKS, DODGE_FRAMES, MAX_ENERGY, MAX_HP,
                                        SPECIAL_COST, Fighter, Projectile, State)

BG_TOP = (28, 22, 46)
BG_BOTTOM = (88, 46, 60)
FLOOR = (44, 36, 40)
FLOOR_LINE = (120, 90, 80)
WHITE = (240, 240, 240)
GOLD = (245, 200, 70)
RED = (220, 60, 60)
GREEN = (90, 210, 110)
BLUE = (80, 160, 240)
GREY = (120, 120, 130)


class Renderer:
    def __init__(self, screen: pygame.Surface, ground_y: int):
        self.screen = screen
        self.w, self.h = screen.get_size()
        self.ground_y = ground_y
        pygame.font.init()
        self.font_big = pygame.font.SysFont("arial", 64, bold=True)
        self.font_mid = pygame.font.SysFont("arial", 30, bold=True)
        self.font_small = pygame.font.SysFont("arial", 18)
        self._background = self._make_background()

    # ------------------------------------------------------------ background
    def _make_background(self) -> pygame.Surface:
        surf = pygame.Surface((self.w, self.h))
        for y in range(self.ground_y):
            t = y / self.ground_y
            color = [int(BG_TOP[i] * (1 - t) + BG_BOTTOM[i] * t) for i in range(3)]
            pygame.draw.line(surf, color, (0, y), (self.w, y))
        # Distant pillars for some depth.
        for i in range(7):
            x = 60 + i * (self.w - 120) // 6
            pygame.draw.rect(surf, (52, 36, 62), (x - 18, self.ground_y - 300, 36, 300))
            pygame.draw.rect(surf, (70, 48, 80), (x - 26, self.ground_y - 310, 52, 14))
        pygame.draw.rect(surf, FLOOR, (0, self.ground_y, self.w, self.h - self.ground_y))
        for i in range(-10, 30):
            x = self.w / 2 + i * 70
            pygame.draw.line(surf, FLOOR_LINE, (self.w / 2 + (x - self.w / 2) * 0.35, self.ground_y),
                             (x, self.h), 1)
        pygame.draw.line(surf, FLOOR_LINE, (0, self.ground_y), (self.w, self.ground_y), 3)
        return surf

    def draw_background(self) -> None:
        self.screen.blit(self._background, (0, 0))

    # --------------------------------------------------------------- fighter
    def draw_fighter(self, f: Fighter) -> None:
        s = self.screen
        d = f.facing
        x, gy = f.x, f.ground_y
        color = f.color
        dark = tuple(int(c * 0.6) for c in color)
        if f.state == State.HIT and f.state_frame % 4 < 2:
            color = WHITE

        crouch = 0.0
        if f.state == State.DODGE:
            t = f.state_frame / DODGE_FRAMES
            crouch = math.sin(min(1.0, t) * math.pi) * 60
        bob = math.sin(f.anim_time * 0.12) * 3 if f.state in (State.IDLE, State.BLOCK) else 0

        hip = (x, gy - 95 + crouch * 0.6 + bob)
        neck = (x + d * 4, gy - 165 + crouch + bob)
        head = (neck[0] + d * 2, neck[1] - 22)

        # Legs
        step = math.sin(f.anim_time * 0.35) * 14 if f.state == State.WALK else 0
        front_foot = (x + d * (24 + step), gy)
        back_foot = (x - d * (24 + step), gy)
        if f.state == State.KICK:
            ext = self._extension(f)
            front_foot = (x + d * (30 + 90 * ext), gy - 60 - 30 * ext)
        knee_f = ((hip[0] + front_foot[0]) / 2 + d * 10, (hip[1] + front_foot[1]) / 2 - 5)
        knee_b = ((hip[0] + back_foot[0]) / 2 - d * 4, (hip[1] + back_foot[1]) / 2 - 5)
        for a, b, c in ((hip, knee_b, back_foot), (hip, knee_f, front_foot)):
            pygame.draw.line(s, dark, a, b, 14)
            pygame.draw.line(s, dark, b, c, 12)

        # Torso
        pygame.draw.line(s, color, hip, neck, 26)
        pygame.draw.circle(s, color, (int(hip[0]), int(hip[1])), 13)

        # Arms
        shoulder = (neck[0], neck[1] + 10)
        guard = (neck[0] + d * 30, neck[1] + 4)
        rear_guard = (neck[0] + d * 18, neck[1] + 16)
        front_hand, rear_hand = guard, rear_guard
        if f.state == State.PUNCH:
            ext = self._extension(f)
            front_hand = (shoulder[0] + d * (30 + 70 * ext), shoulder[1] - 4)
        elif f.state == State.SPECIAL:
            ext = self._extension(f)
            front_hand = (shoulder[0] + d * (30 + 40 * ext), shoulder[1] + 4)
            rear_hand = (shoulder[0] + d * (26 + 40 * ext), shoulder[1] + 14)
        elif f.state == State.BLOCK:
            front_hand = (neck[0] + d * 26, neck[1] - 22)
            rear_hand = (neck[0] + d * 22, neck[1] - 4)
        for hand in (rear_hand, front_hand):
            elbow = ((shoulder[0] + hand[0]) / 2 - d * 4, (shoulder[1] + hand[1]) / 2 + 12)
            pygame.draw.line(s, dark if hand is rear_hand else color, shoulder, elbow, 10)
            pygame.draw.line(s, dark if hand is rear_hand else color, elbow, hand, 9)
            pygame.draw.circle(s, (230, 190, 160), (int(hand[0]), int(hand[1])), 9)

        # Head with a headband
        pygame.draw.circle(s, (230, 190, 160), (int(head[0]), int(head[1])), 19)
        pygame.draw.line(s, color, (head[0] - 19, head[1] - 6), (head[0] + 19, head[1] - 6), 6)
        pygame.draw.line(s, color, (head[0] - d * 18, head[1] - 6),
                         (head[0] - d * 34, head[1] + 2 + bob * 2), 4)
        pygame.draw.circle(s, (30, 30, 30), (int(head[0] + d * 8), int(head[1] + 1)), 3)

        if f.state == State.BLOCK:
            pygame.draw.arc(s, BLUE, (x + d * 20 - 40, gy - 200, 80, 140),
                            -math.pi / 2 if d > 0 else math.pi / 2,
                            math.pi / 2 if d > 0 else 3 * math.pi / 2, 3)

    @staticmethod
    def _extension(f: Fighter) -> float:
        """0 -> 1 -> 0 over the attack, peaking during active frames."""
        spec = ATTACKS[f.state]
        t = f.state_frame
        if t < spec.startup:
            return t / max(1, spec.startup)
        if t < spec.startup + spec.active:
            return 1.0
        return max(0.0, 1 - (t - spec.startup - spec.active) / max(1, spec.recovery))

    def draw_projectile(self, p: Projectile) -> None:
        r = p.radius
        for i, alpha in enumerate((1.0, 0.6, 0.3)):
            pygame.draw.circle(self.screen, (int(255 * alpha), int(160 * alpha), 40),
                               (int(p.x - p.direction * i * 14), int(p.y)), r - i * 4)
        pygame.draw.circle(self.screen, (255, 240, 180), (int(p.x), int(p.y)), r // 2)

    # ------------------------------------------------------------------- HUD
    def draw_hud(self, left: Fighter, right: Fighter, time_left: float) -> None:
        bar_w, bar_h, top = self.w // 2 - 110, 26, 22
        self._bar(30, top, bar_w, bar_h, left.hp / MAX_HP, GREEN, flip=False)
        self._bar(self.w - 30 - bar_w, top, bar_w, bar_h, right.hp / MAX_HP, GREEN, flip=True)
        self._bar(30, top + 32, bar_w // 2, 8, left.energy / MAX_ENERGY,
                  GOLD if left.energy >= SPECIAL_COST else GREY, flip=False)
        self._bar(self.w - 30 - bar_w // 2, top + 32, bar_w // 2, 8, right.energy / MAX_ENERGY,
                  GOLD if right.energy >= SPECIAL_COST else GREY, flip=True)
        self._text(left.name, self.font_small, (30, top + 44))
        name = self.font_small.render(right.name, True, WHITE)
        self.screen.blit(name, (self.w - 30 - name.get_width(), top + 44))
        timer = self.font_mid.render(f"{max(0, int(time_left)):02d}", True, GOLD)
        self.screen.blit(timer, (self.w // 2 - timer.get_width() // 2, top - 4))

    def _bar(self, x, y, w, h, ratio, color, flip):
        pygame.draw.rect(self.screen, (30, 20, 20), (x - 2, y - 2, w + 4, h + 4))
        pygame.draw.rect(self.screen, RED, (x, y, w, h))
        fill = int(w * max(0.0, min(1.0, ratio)))
        pygame.draw.rect(self.screen, color, (x + (w - fill if flip else 0), y, fill, h))

    def _text(self, text, font, pos, color=WHITE):
        self.screen.blit(font.render(text, True, color), pos)

    def draw_center_message(self, title: str, subtitle: str = "", color=GOLD) -> None:
        t = self.font_big.render(title, True, color)
        self.screen.blit(t, (self.w // 2 - t.get_width() // 2, self.h // 2 - 110))
        if subtitle:
            st = self.font_small.render(subtitle, True, WHITE)
            self.screen.blit(st, (self.w // 2 - st.get_width() // 2, self.h // 2 - 30))

    def draw_status(self, lines: list[str], progress: float | None = None) -> None:
        """Input status at the bottom left (gesture, phase...)."""
        y = self.h - 26 * len(lines) - 10
        for line in lines:
            self._text(line, self.font_small, (16, y), (220, 220, 230))
            y += 24
        if progress is not None and progress < 1.0:
            pygame.draw.rect(self.screen, GREY, (16, self.h - 8, 300, 4))
            pygame.draw.rect(self.screen, GOLD, (16, self.h - 8, int(300 * progress), 4))

    def draw_thumbnail(self, bgr: np.ndarray | None, width: int = 240) -> None:
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        height = int(width * h / w)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        surf = pygame.transform.smoothscale(surf, (width, height))
        x, y = self.w - width - 16, self.h - height - 16
        pygame.draw.rect(self.screen, (20, 20, 20), (x - 3, y - 3, width + 6, height + 6))
        self.screen.blit(surf, (x, y))
