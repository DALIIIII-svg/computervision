"""Bot opponent: a finite-state machine with human-like reaction time.

States:
    APPROACH  walk toward the player
    ATTACK    choose punch / kick / special depending on distance
    RETREAT   step back after attacking
    DEFEND    react to the player's attack (block or dodge), after a reaction delay
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from kombat_vision.actions import Actions
from kombat_vision.game.fighter import ATTACKS, SPECIAL_COST, Fighter, Projectile, State


@dataclass(frozen=True)
class Difficulty:
    reaction_frames: int      # delay before reacting to an attack
    defend_prob: float        # chance to defend against a seen attack
    aggression: float         # chance per decision to attack when in range
    special_prob: float       # chance to throw a special from far away
    think_every: int          # frames between decisions


DIFFICULTIES = {
    "easy": Difficulty(reaction_frames=22, defend_prob=0.25, aggression=0.25, special_prob=0.05, think_every=14),
    "normal": Difficulty(reaction_frames=14, defend_prob=0.5, aggression=0.45, special_prob=0.10, think_every=9),
    "hard": Difficulty(reaction_frames=8, defend_prob=0.75, aggression=0.7, special_prob=0.18, think_every=5),
}


class BotState(str, Enum):
    APPROACH = "approach"
    ATTACK = "attack"
    RETREAT = "retreat"
    DEFEND = "defend"
    WAIT = "wait"


class Bot:
    def __init__(self, fighter: Fighter, difficulty: str = "normal", seed: int | None = None):
        self.fighter = fighter
        self.diff = DIFFICULTIES.get(difficulty, DIFFICULTIES["normal"])
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.state = BotState.WAIT
        self.timer = 30
        self._pending_defense: tuple[int, str] | None = None   # (frames left, "block"/"dodge")
        self._seen_attack_id = None
        self._think = 0

    def get_actions(self, player: Fighter, projectiles: list[Projectile]) -> Actions:
        me = self.fighter
        a = Actions()
        if not me.alive or not player.alive:
            return a
        distance = abs(player.x - me.x)
        toward = 1 if player.x > me.x else -1

        self._watch_threats(player, projectiles, distance)

        # Execute a scheduled defense once the reaction delay is over.
        if self._pending_defense is not None:
            frames, kind = self._pending_defense
            if frames <= 0:
                self.state = BotState.DEFEND
                self.timer = 22
                self._defense_kind = kind
                self._pending_defense = None
            else:
                self._pending_defense = (frames - 1, kind)

        if self.state == BotState.DEFEND:
            self.timer -= 1
            if self._defense_kind == "dodge":
                a.dodge = True
            else:
                a.block = True
            if self.timer <= 0:
                self.state = BotState.APPROACH
            return a

        self.timer -= 1
        self._think -= 1
        if self._think <= 0:
            self._think = self.diff.think_every
            self._decide(distance, me, player)

        if self.state == BotState.APPROACH:
            a.move_x = toward if distance > ATTACKS[State.PUNCH].reach - 10 else 0
        elif self.state == BotState.RETREAT:
            a.move_x = -toward
            if self.timer <= 0:
                self.state = BotState.APPROACH
        elif self.state == BotState.ATTACK:
            if distance <= ATTACKS[State.PUNCH].reach:
                a.punch = True
            elif distance <= ATTACKS[State.KICK].reach:
                a.kick = True
            else:
                a.special = True
            self.state = BotState.RETREAT if self.rng.random() < 0.4 else BotState.APPROACH
            self.timer = self.rng.randint(10, 30)
        elif self.state == BotState.WAIT and self.timer <= 0:
            self.state = BotState.APPROACH
        return a

    def _decide(self, distance: float, me: Fighter, player: Fighter) -> None:
        if self.state in (BotState.RETREAT, BotState.WAIT):
            return
        in_range = distance <= ATTACKS[State.KICK].reach
        if in_range and self.rng.random() < self.diff.aggression:
            self.state = BotState.ATTACK
        elif distance > 300 and me.energy >= SPECIAL_COST and self.rng.random() < self.diff.special_prob:
            self.state = BotState.ATTACK
        else:
            self.state = BotState.APPROACH

    def _watch_threats(self, player: Fighter, projectiles: list[Projectile], distance: float) -> None:
        """Notice the player's attacks (once each) and maybe schedule a defense."""
        threat_id, kind = None, None
        spec = player.attack
        if spec is not None and player.attack_phase() == "startup" and distance <= spec.reach + 40:
            threat_id = (player.state, player.anim_time - player.state_frame)
            kind = "dodge" if spec.high and self.rng.random() < 0.5 else "block"
        for p in projectiles:
            if p.owner is player and abs(p.x - self.fighter.x) < 260:
                threat_id, kind = id(p), "dodge"
        if threat_id is None or threat_id == self._seen_attack_id or self._pending_defense:
            return
        self._seen_attack_id = threat_id
        if self.rng.random() < self.diff.defend_prob:
            jitter = self.rng.randint(-3, 3)
            self._pending_defense = (max(1, self.diff.reaction_frames + jitter), kind)
