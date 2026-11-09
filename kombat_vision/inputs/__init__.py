"""Input sources. Each one produces `Actions` for the game."""

from kombat_vision.config import Config
from kombat_vision.inputs.base import InputSource

INPUT_NAMES = ("keyboard", "vision", "mediapipe")


def create_input(name: str, cfg: Config) -> InputSource:
    if name == "keyboard":
        from kombat_vision.inputs.keyboard import KeyboardInput
        return KeyboardInput()
    if name == "vision":
        from kombat_vision.inputs.vision import VisionInput
        return VisionInput(cfg)
    if name == "mediapipe":
        from kombat_vision.inputs.mediapipe_baseline import MediaPipeInput
        return MediaPipeInput(cfg)
    raise ValueError(f"Unknown input '{name}'. Choose from {INPUT_NAMES}")
