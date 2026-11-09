"""Vision lab: run the perception pipeline alone (no game), to tune and test it.

    python scripts/vision_lab.py
    python scripts/vision_lab.py --background codebook
    python scripts/vision_lab.py --video data/raw/test.mp4     # replay a recorded video

Keys: b relearn background, c recalibrate skin, n skip calibration,
      p pause, r start/stop recording a video to data/raw/, q quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kombat_vision.config import load_config  # noqa: E402
from kombat_vision.vision.camera import Camera  # noqa: E402
from kombat_vision.vision.debug_view import HELP_TEXT, compose  # noqa: E402
from kombat_vision.vision.pipeline import VisionPipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--background", choices=("mog2", "codebook", "none"))
    p.add_argument("--video", help="use a video file instead of the webcam")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.background:
        cfg.background.method = args.background
    pipeline = VisionPipeline(cfg)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        read = lambda: cap.read()[1]  # noqa: E731
        camera = None
    else:
        camera = Camera(cfg.camera).start()
        read = lambda: camera.read()[1]  # noqa: E731

    print(HELP_TEXT + "  [p] pause  [r] record")
    paused, writer, last = False, None, None
    try:
        while True:
            if not paused:
                frame = read()
                if frame is None:
                    if args.video:
                        break
                    time.sleep(0.005)
                    continue
                if writer is not None:
                    writer.write(frame)
                last = compose(pipeline.process(frame), pipeline.skin, cfg.gestures.move_zones)
            if last is not None:
                if writer is not None:
                    cv2.circle(last, (last.shape[1] - 20, 20), 8, (0, 0, 255), -1)
                cv2.imshow("Kombat Vision - lab", last)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                pipeline.restart_background()
            elif key == ord("c"):
                pipeline.restart_calibration()
            elif key == ord("n"):
                pipeline.skip_calibration()
            elif key == ord("p"):
                paused = not paused
            elif key == ord("r") and frame is not None:
                if writer is None:
                    out = ROOT / "data" / "raw" / f"session_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                    print(f"recording to {out}")
                else:
                    writer.release()
                    writer = None
                    print("recording stopped")
    finally:
        if writer is not None:
            writer.release()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
