"""Kombat Vision - entry point.

Examples:
    python main.py                          # keyboard only
    python main.py --input vision --debug   # our CV pipeline + debug window
    python main.py --input mediapipe        # MediaPipe baseline
    python main.py --input vision --background codebook --difficulty hard
    python main.py --input vision --video data/raw/session.mp4 --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kombat_vision.config import load_config
from kombat_vision.inputs import INPUT_NAMES, create_input

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fighting game controlled by hand gestures.")
    p.add_argument("--input", choices=INPUT_NAMES, default="keyboard", help="how the player is controlled")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
    p.add_argument("--debug", action="store_true", help="show the pipeline debug window")
    p.add_argument("--camera", type=int, help="camera index (overrides the config)")
    p.add_argument("--video", help="play a recorded video instead of the webcam")
    p.add_argument("--background", choices=("mog2", "codebook", "none"), help="background subtraction method")
    p.add_argument("--difficulty", choices=("easy", "normal", "hard"), help="bot difficulty")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.camera is not None:
        cfg.camera.index = args.camera
    if args.video:
        cfg.camera.index = args.video
        cfg.camera.mirror = False   # recordings from vision_lab are already mirrored
    if args.background:
        cfg.background.method = args.background
    if args.difficulty:
        cfg.game.bot_difficulty = args.difficulty

    from kombat_vision.game.engine import Game  # imported late: pygame prints a banner
    source = create_input(args.input, cfg)
    Game(cfg, source, debug=args.debug).run()


if __name__ == "__main__":
    main()
