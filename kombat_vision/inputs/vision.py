"""Our classical computer vision pipeline as a game input."""

from __future__ import annotations

import numpy as np
import pygame

from kombat_vision.config import Config
from kombat_vision.inputs.camera_input import CameraInput
from kombat_vision.vision.debug_view import compose
from kombat_vision.vision.pipeline import Phase, PipelineResult, VisionPipeline


class VisionInput(CameraInput):
    name = "vision (classical CV)"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.pipeline = VisionPipeline(cfg)

    def process(self, frame: np.ndarray) -> PipelineResult:
        return self.pipeline.process(frame)

    def handle_command(self, key: int) -> None:
        if key == pygame.K_b:
            self.pipeline.restart_background()
        elif key == pygame.K_c:
            self.pipeline.restart_calibration()
        elif key == pygame.K_n:
            self.pipeline.skip_calibration()

    # ---------------------------------------------------------------- status
    @property
    def ready(self) -> bool:
        res = self.latest()
        return res is not None and res.phase == Phase.RUNNING

    @property
    def message(self) -> str:
        if self.error:
            return self.error
        res = self.latest()
        if res is None:
            return "Opening camera..."
        if res.phase == Phase.CALIBRATE:
            return res.message + "  (N = skip, use default skin ranges)"
        return res.message

    @property
    def progress(self) -> float | None:
        res = self.latest()
        return res.progress if res is not None else None

    def status_lines(self) -> list[str]:
        res = self.latest()
        if res is None:
            return []
        return [f"gesture: {res.gesture.value}   vision: {res.fps:.0f} FPS, {res.timings.get('total', 0):.0f} ms",
                "B relearn background   C recalibrate skin"]

    def thumbnail(self) -> np.ndarray | None:
        res = self.latest()
        if res is None or res.frame is None:
            return None
        from kombat_vision.vision.debug_view import draw_annotations
        return draw_annotations(res, self.pipeline.skin, self.cfg.gestures.move_zones)

    def debug_image(self, result: PipelineResult) -> np.ndarray | None:
        return compose(result, self.pipeline.skin, self.cfg.gestures.move_zones)
