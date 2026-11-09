"""Fighter logic: states, attacks, damage. No drawing here (see renderer.py).

All timings are in frames (the game runs at a fixed 60 FPS).

Rules:
  - punch: fast, high          -> avoided by dodge, reduced by block
  - kick:  slower, low, longer -> NOT avoided by dodge (you duck into it), reduced by block
  - special: projectile, costs energy, high -> avoided by dodge
  - block: held, cannot move or attack, takes 20% damage
  - dodge: short duck, cannot act until it ends
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kombat_vision.actions import Actions


class State(str, Enum):
    IDLE = "idle"
    WALK = "walk"
    PUNCH = "punch"
    KICK = "kick"
    SPECIAL = "special"
    BLOCK = "block"
    DODGE = "dodge"
    HIT = "hit"
    KO = "ko"


@dataclass(frozen=True)
class AttackSpec:
    startup: int
    active: int
    recovery: int
    damage: float
    reach: int
    high: bool          # high attacks are avoided by dodge

    @property
    def total(self) -> int:
        return self.startup + self.active + self.recovery


ATTACKS = {
    State.PUNCH: AttackSpec(startup=5, active=5, recovery=10, damage=7, reach=105, high=True),
    State.KICK: AttackSpec(startup=10, active=6, recovery=16, damage=11, reach=135, high=False),
    State.SPECIAL: AttackSpec(startup=14, active=1, recovery=20, damage=14, reach=0, high=True),
}

MAX_HP = 100.0
MAX_ENERGY = 100.0
SPECIAL_COST = 50.0
WALK_SPEED = 4.5
DODGE_FRAMES = 26
HIT_STUN = 14
BLOCK_DAMAGE_RATIO = 0.2
BODY_WIDTH = 60


@dataclass
class Projectile:
    x: float
    y: float
    direction: int
    owner: "Fighter"
    damage: float
    speed: float = 9.0
    alive: bool = True
    radius: int = 18

    def update(self, arena_width: int) -> None:
        self.x += self.direction * self.speed
        if self.x < -50 or self.x > arena_width + 50:
            self.alive = False


class Fighter:
    def __init__(self, name: str, x: float, ground_y: float, facing: int, color: tuple):
        self.name = name
        self.start_x = x
        self.ground_y = ground_y
        self.color = color
        self.start_facing = facing
        self.reset()

    def reset(self) -> None:
        self.x = self.start_x
        self.facing = self.start_facing
        self.hp = MAX_HP
        self.energy = 30.0
        self.state = State.IDLE
        self.state_frame = 0
        self.attack_connected = False
        self.knockback = 0.0
        self.anim_time = 0          # increases every frame, used by the renderer

    # -------------------------------------------------------------- helpers
    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def busy(self) -> bool:
        """True while an action is playing and new input is ignored."""
        return self.state in (State.PUNCH, State.KICK, State.SPECIAL, State.DODGE, State.HIT, State.KO)

    @property
    def attack(self) -> AttackSpec | None:
        return ATTACKS.get(self.state)

    def attack_phase(self) -> str | None:
        """'startup', 'active' or 'recovery' while attacking."""
        spec = self.attack
        if spec is None:
            return None
        if self.state_frame < spec.startup:
            return "startup"
        if self.state_frame < spec.startup + spec.active:
            return "active"
        return "recovery"

    def _set_state(self, state: State) -> None:
        self.state = state
        self.state_frame = 0
        self.attack_connected = False

    # ----------------------------------------------------------- per frame
    def update(self, actions: Actions, opponent: "Fighter", arena: tuple[int, int],
               projectiles: list[Projectile]) -> None:
        self.anim_time += 1
        self.state_frame += 1
        self.energy = min(MAX_ENERGY, self.energy + 0.04)

        if self.state == State.KO:
            return

        # Knockback slides the fighter back after being hit.
        if abs(self.knockback) > 0.2:
            self.x += self.knockback
            self.knockback *= 0.8

        # Finish timed states.
        spec = self.attack
        if spec is not None:
            if self.state == State.SPECIAL and self.state_frame == spec.startup:
                self._spawn_projectile(projectiles)
            if self.state_frame >= spec.total:
                self._set_state(State.IDLE)
        elif self.state == State.DODGE and self.state_frame >= DODGE_FRAMES:
            self._set_state(State.IDLE)
        elif self.state == State.HIT and self.state_frame >= HIT_STUN:
            self._set_state(State.IDLE)

        if not self.busy:
            self.facing = 1 if opponent.x > self.x else -1
            self._handle_input(actions, projectiles)

        # Stay inside the arena and do not walk through the opponent.
        left, right = 40, arena[0] - 40
        self.x = max(left, min(right, self.x))
        gap = BODY_WIDTH
        if abs(self.x - opponent.x) < gap:
            push = (gap - abs(self.x - opponent.x)) / 2
            self.x += -push if self.x < opponent.x else push

    def _handle_input(self, a: Actions, projectiles: list[Projectile]) -> None:
        if a.dodge:
            self._set_state(State.DODGE)
        elif a.punch:
            self._set_state(State.PUNCH)
        elif a.kick:
            self._set_state(State.KICK)
        elif a.special and self.energy >= SPECIAL_COST:
            self.energy -= SPECIAL_COST
            self._set_state(State.SPECIAL)
        elif a.block:
            if self.state != State.BLOCK:
                self._set_state(State.BLOCK)
        elif a.move_x != 0:
            if self.state != State.WALK:
                self._set_state(State.WALK)
            self.x += a.move_x * WALK_SPEED
        elif self.state != State.IDLE:
            self._set_state(State.IDLE)

    def _spawn_projectile(self, projectiles: list[Projectile]) -> None:
        spec = ATTACKS[State.SPECIAL]
        projectiles.append(Projectile(self.x + self.facing * 50, self.ground_y - 120,
                                      self.facing, self, spec.damage))

    # ---------------------------------------------------------- combat
    def can_hit(self, opponent: "Fighter") -> bool:
        """True if this fighter's active attack frames reach the opponent."""
        spec = self.attack
        if spec is None or self.state == State.SPECIAL or self.attack_connected:
            return False
        if self.attack_phase() != "active":
            return False
        dx = (opponent.x - self.x) * self.facing
        return 0 < dx <= spec.reach

    def receive_hit(self, damage: float, high: bool, from_x: float) -> str:
        """Apply a hit. Returns 'dodged', 'blocked' or 'hit'."""
        if self.state == State.KO:
            return "dodged"
        if self.state == State.DODGE and high:
            return "dodged"
        direction = 1 if self.x > from_x else -1
        if self.state == State.BLOCK:
            self.hp = max(0.0, self.hp - damage * BLOCK_DAMAGE_RATIO)
            self.knockback = direction * 3
            self.energy = min(MAX_ENERGY, self.energy + 4)
            result = "blocked"
        else:
            self.hp = max(0.0, self.hp - damage)
            self.knockback = direction * 7
            self.energy = min(MAX_ENERGY, self.energy + 8)
            self._set_state(State.HIT)
            result = "hit"
        if self.hp <= 0:
            self._set_state(State.KO)
        return result
