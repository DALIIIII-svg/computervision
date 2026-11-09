import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from kombat_vision.actions import Actions  # noqa: E402
from kombat_vision.game.fighter import ATTACKS, Fighter, State  # noqa: E402


def make_pair(distance=90):
    a = Fighter("A", 300, 500, 1, (0, 0, 255))
    b = Fighter("B", 300 + distance, 500, -1, (255, 0, 0))
    return a, b


def run_attack(attacker, defender, defender_actions=Actions()):
    projectiles = []
    for _ in range(ATTACKS[State.KICK].total + 2):
        if attacker.can_hit(defender):
            attacker.attack_connected = True
            spec = attacker.attack
            defender.receive_hit(spec.damage, spec.high, attacker.x)
        attacker.update(Actions(), defender, (1000, 600), projectiles)
        defender.update(defender_actions, attacker, (1000, 600), projectiles)


def test_punch_hits_once():
    a, b = make_pair()
    a.update(Actions(punch=True), b, (1000, 600), [])
    run_attack(a, b)
    assert b.hp == 100 - ATTACKS[State.PUNCH].damage


def test_dodge_avoids_punch_but_not_kick():
    a, b = make_pair()
    b.update(Actions(dodge=True), a, (1000, 600), [])
    a.update(Actions(punch=True), b, (1000, 600), [])
    run_attack(a, b)
    assert b.hp == 100

    a, b = make_pair(110)
    b.update(Actions(dodge=True), a, (1000, 600), [])
    a.update(Actions(kick=True), b, (1000, 600), [])
    run_attack(a, b)
    assert b.hp < 100


def test_block_reduces_damage():
    a, b = make_pair()
    a.update(Actions(punch=True), b, (1000, 600), [])
    run_attack(a, b, Actions(block=True))
    assert 100 - ATTACKS[State.PUNCH].damage < b.hp < 100


def test_game_runs_headless_with_keyboard():
    from kombat_vision.config import Config
    from kombat_vision.game.engine import Game
    from kombat_vision.inputs.keyboard import KeyboardInput

    game = Game(Config(), KeyboardInput(), headless_frames=400)
    game.run()
    assert game.enemy.state_frame >= 0       # bot ran without crashing
