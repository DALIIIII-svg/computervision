"""Background subtraction: MOG2 (OpenCV) and Codebook (our NumPy implementation)."""

from kombat_vision.config import BackgroundConfig
from kombat_vision.vision.background.base import BackgroundModel, NoBackground
from kombat_vision.vision.background.codebook import CodebookBackground
from kombat_vision.vision.background.mog import Mog2Background


def create_background_model(cfg: BackgroundConfig) -> BackgroundModel:
    method = cfg.method.lower()
    if method == "mog2":
        return Mog2Background(cfg.mog2, cfg.warmup_frames)
    if method == "codebook":
        return CodebookBackground(cfg.codebook, cfg.warmup_frames)
    if method == "none":
        return NoBackground()
    raise ValueError(f"Unknown background method: {cfg.method} (use mog2, codebook or none)")


__all__ = ["BackgroundModel", "NoBackground", "Mog2Background", "CodebookBackground",
           "create_background_model"]
