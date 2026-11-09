"""Configuration: typed defaults + optional YAML overrides.

Usage:
    cfg = load_config("configs/default.yaml")
    cfg.background.method  # "mog2"
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True
    use_directshow: bool = True


@dataclass
class PreprocessingConfig:
    scale: float = 0.5
    blur_ksize: int = 5


@dataclass
class Mog2Config:
    history: int = 300
    var_threshold: float = 25
    detect_shadows: bool = True
    learning_rate: float = -1
    learning_rate_after: float = 0.0005


@dataclass
class CodebookConfig:
    max_codewords: int = 5
    color_epsilon: float = 12.0
    alpha: float = 0.6
    beta: float = 1.25


@dataclass
class BackgroundConfig:
    method: str = "mog2"
    warmup_frames: int = 60
    mog2: Mog2Config = field(default_factory=Mog2Config)
    codebook: CodebookConfig = field(default_factory=CodebookConfig)


@dataclass
class SkinConfig:
    calibration_frames: int = 45
    roi: list = field(default_factory=lambda: [0.42, 0.30, 0.16, 0.24])
    margin_cr: int = 6
    margin_cb: int = 6
    min_y: int = 40
    default_cr: list = field(default_factory=lambda: [133, 173])
    default_cb: list = field(default_factory=lambda: [77, 127])
    morph_kernel: int = 5


@dataclass
class FaceConfig:
    enabled: bool = True
    scale_factor: float = 1.15
    min_neighbors: int = 5
    min_size: int = 40
    detect_every: int = 2
    mask_padding: float = 0.25


@dataclass
class HandConfig:
    min_area_ratio: float = 0.008
    use_region_growing: bool = True
    region_grow_tolerance: int = 10
    use_canny_split: bool = True
    canny_low: int = 40
    canny_high: int = 110
    wrist_cut: bool = True
    wrist_cut_factor: float = 1.7


@dataclass
class FingersConfig:
    defect_max_angle: float = 90
    defect_min_depth: float = 0.35
    extended_ratio: float = 1.75


@dataclass
class TrackingConfig:
    process_noise: float = 1.0
    measurement_noise: float = 8.0
    max_lost_frames: int = 8


@dataclass
class DodgeConfig:
    velocity_threshold: float = 0.10
    drop_threshold: float = 0.55
    cooldown_s: float = 0.8


@dataclass
class GesturesConfig:
    stable_frames: int = 4
    hold_repeat_s: float = 0.7
    move_zones: list = field(default_factory=lambda: [0.33, 0.67])
    mapping: dict = field(default_factory=lambda: {
        "one": "punch", "four": "kick", "palm": "block",
        "two": "special", "fist": "none", "three": "none",
    })


@dataclass
class GameConfig:
    width: int = 1100
    height: int = 620
    fps: int = 60
    round_time_s: int = 99
    bot_difficulty: str = "normal"
    show_camera_thumbnail: bool = True
    keyboard_always_on: bool = True


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    skin: SkinConfig = field(default_factory=SkinConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    fingers: FingersConfig = field(default_factory=FingersConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    dodge: DodgeConfig = field(default_factory=DodgeConfig)
    gestures: GesturesConfig = field(default_factory=GesturesConfig)
    game: GameConfig = field(default_factory=GameConfig)


def _apply(target: Any, values: dict, path: str = "") -> None:
    """Recursively copy YAML values onto a dataclass, rejecting unknown keys."""
    known = {f.name for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key: {path}{key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, f"{path}{key}.")
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _apply(cfg, data)
    return cfg
