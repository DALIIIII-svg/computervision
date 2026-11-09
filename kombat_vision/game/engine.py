"""Game loop: input -> fighters -> collisions -> rendering, at a fixed 60 FPS."""

from __future__ import annotations

import pygame

from kombat_vision.actions import Actions
from kombat_vision.config import Config
from kombat_vision.game.bot import Bot
from kombat_vision.game.fighter import Fighter, Projectile, State
from kombat_vision.game.renderer import Renderer
from kombat_vision.inputs.base import InputSource
from kombat_vision.inputs.keyboard import KeyboardInput, KEY_HELP


class Game:
    def __init__(self, cfg: Config, source: InputSource, debug: bool = False,
                 headless_frames: int | None = None):
        self.cfg = cfg
        self.source = source
        self.debug = debug
        self.headless_frames = headless_frames      # run N frames without a window (tests)
        self.keyboard = source if isinstance(source, KeyboardInput) else (
            KeyboardInput() if cfg.game.keyboard_always_on else None)

        pygame.init()
        g = cfg.game
        flags = pygame.HIDDEN if headless_frames else 0
        self.screen = pygame.display.set_mode((g.width, g.height), flags)
        pygame.display.set_caption("Kombat Vision")
        self.clock = pygame.time.Clock()
        self.ground_y = int(g.height * 0.86)
        self.renderer = Renderer(self.screen, self.ground_y)
        self.player = Fighter("YOU", g.width * 0.3, self.ground_y, 1, (70, 130, 220))
        self.enemy = Fighter("BOT", g.width * 0.7, self.ground_y, -1, (210, 70, 70))
        self.bot = Bot(self.enemy, g.bot_difficulty)
        self.projectiles: list[Projectile] = []
        self.new_round()

    def new_round(self) -> None:
        self.player.reset()
        self.enemy.reset()
        self.bot.reset()
        self.projectiles.clear()
        self.frames_left = self.cfg.game.round_time_s * self.cfg.game.fps
        self.winner: str | None = None
        self.intro_frames = 90

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        self.source.start()
        frame = 0
        try:
            running = True
            while running:
                running = self._handle_events()
                self._update()
                self._render()
                if self.debug:
                    self.source.show_debug()
                pygame.display.flip()
                self.clock.tick(self.cfg.game.fps)
                frame += 1
                if self.headless_frames and frame >= self.headless_frames:
                    break
        finally:
            self.source.close()
            pygame.quit()

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self.new_round()
                self.source.on_key(event.key)
            if self.keyboard is not None:
                self.keyboard.handle_event(event)
        return True

    def _player_actions(self) -> Actions:
        actions = self.source.poll()
        if self.keyboard is not None and self.keyboard is not self.source:
            actions = actions.merge(self.keyboard.poll())
        return actions

    def _update(self) -> None:
        arena = (self.cfg.game.width, self.cfg.game.height)
        player_actions = self._player_actions()     # always poll, so events do not pile up
        if not self.source.ready or self.winner or self.intro_frames > 0:
            self.intro_frames = max(0, self.intro_frames - (1 if self.source.ready else 0))
            player_actions, bot_actions = Actions(), Actions()
            if self.winner:
                self.player.update(player_actions, self.enemy, arena, self.projectiles)
                self.enemy.update(bot_actions, self.player, arena, self.projectiles)
            return

        bot_actions = self.bot.get_actions(self.player, self.projectiles)
        self.player.update(player_actions, self.enemy, arena, self.projectiles)
        self.enemy.update(bot_actions, self.player, arena, self.projectiles)
        self._resolve_hits()

        for p in self.projectiles:
            p.update(arena[0])
        self.projectiles = [p for p in self.projectiles if p.alive]

        self.frames_left -= 1
        if not self.player.alive:
            self.winner = self.enemy.name
        elif not self.enemy.alive:
            self.winner = self.player.name
        elif self.frames_left <= 0:
            self.winner = self.player.name if self.player.hp >= self.enemy.hp else self.enemy.name

    def _resolve_hits(self) -> None:
        for attacker, defender in ((self.player, self.enemy), (self.enemy, self.player)):
            if attacker.can_hit(defender):
                attacker.attack_connected = True
                spec = attacker.attack
                defender.receive_hit(spec.damage, spec.high, attacker.x)
        for p in self.projectiles:
            target = self.enemy if p.owner is self.player else self.player
            if abs(p.x - target.x) < 34 + p.radius:
                if target.receive_hit(p.damage, True, p.x) != "dodged":
                    p.alive = False

    # ---------------------------------------------------------------- render
    def _render(self) -> None:
        r = self.renderer
        r.draw_background()
        for f in sorted((self.player, self.enemy), key=lambda f: f.state == State.KO, reverse=True):
            r.draw_fighter(f)
        for p in self.projectiles:
            r.draw_projectile(p)
        r.draw_hud(self.player, self.enemy, self.frames_left / self.cfg.game.fps)

        status = [f"input: {self.source.name}"] + self.source.status_lines()
        if self.keyboard is not None:
            status.append(KEY_HELP)
        r.draw_status(status, self.source.progress)
        if self.cfg.game.show_camera_thumbnail:
            r.draw_thumbnail(self.source.thumbnail())

        if not self.source.ready:
            r.draw_center_message("GET READY", self.source.message)
        elif self.winner:
            title = "YOU WIN" if self.winner == self.player.name else "K.O."
            r.draw_center_message(title, "press R to play again, ESC to quit")
        elif self.intro_frames > 0:
            r.draw_center_message("FIGHT!" if self.intro_frames < 45 else "ROUND 1")
