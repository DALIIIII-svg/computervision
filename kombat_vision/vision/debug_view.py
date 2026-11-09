"""Debug view: every pipeline stage side by side, for tuning, the demo and the report.

    +----------------------+----------------------+
    | annotated camera     | foreground (MOG/CB)  |
    +----------------------+----------------------+
    | skin mask            | hand + hull + defects|
    +----------------------+----------------------+
    | timings, FPS, gesture, skin ranges          |
    +---------------------------------------------+
"""

from __future__ import annotations

import cv2
import numpy as np

from kombat_vision.vision.pipeline import Phase, PipelineResult
from kombat_vision.vision.skin import SkinModel

GREEN, RED, BLUE, YELLOW, CYAN, WHITE, MAGENTA = (
    (80, 220, 80), (60, 60, 230), (230, 140, 40), (40, 220, 230), (220, 220, 60), (240, 240, 240), (220, 80, 220))


def _put(img, text, org, color=WHITE, scale=0.45, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _gray3(mask, shape):
    if mask is None:
        return np.zeros(shape, np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def draw_annotations(res: PipelineResult, skin: SkinModel, move_zones=(0.33, 0.67)) -> np.ndarray:
    img = res.frame.copy()
    h, w = img.shape[:2]

    if res.phase == Phase.CALIBRATE:
        x, y, bw, bh = skin.roi_box(img.shape)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), YELLOW, 2)

    if res.phase == Phase.RUNNING:
        for zx in move_zones:
            cv2.line(img, (int(zx * w), 0), (int(zx * w), h), (90, 90, 90), 1)

    if res.face_box is not None:
        x, y, bw, bh = res.face_box
        cv2.rectangle(img, (x, y), (x + bw, y + bh), BLUE, 2)
    head = res.head_track
    if head is not None and head.active:
        cx, cy = (int(v) for v in head.center)
        vx, vy = head.velocity
        cv2.circle(img, (cx, cy), 3, BLUE, -1)
        cv2.arrowedLine(img, (cx, cy), (int(cx + 4 * vx), int(cy + 4 * vy)), BLUE, 2, tipLength=0.3)

    hand = res.hand
    if hand is not None:
        cv2.drawContours(img, [hand.contour], -1, GREEN, 2)
        cv2.circle(img, hand.palm_center, int(hand.palm_radius), MAGENTA, 1)
    if res.fingers is not None:
        if res.fingers.hull is not None:
            cv2.drawContours(img, [res.fingers.hull], -1, CYAN, 1)
        for start, end, far in res.fingers.defects:
            cv2.circle(img, far, 4, RED, -1)
            cv2.line(img, start, far, RED, 1)
            cv2.line(img, end, far, RED, 1)
        if res.fingers.tip is not None:
            cv2.circle(img, res.fingers.tip, 4, YELLOW, -1)
    track = res.hand_track
    if track is not None and track.active:
        cv2.drawMarker(img, (int(track.center[0]), int(track.center[1])), YELLOW, cv2.MARKER_CROSS, 12, 2)

    if res.phase != Phase.RUNNING:
        _put(img, res.message, (8, h - 28), YELLOW, 0.45)
        cv2.rectangle(img, (8, h - 16), (8 + int((w - 16) * res.progress), h - 8), YELLOW, -1)
    else:
        _put(img, f"{res.gesture.value.upper()}", (8, 22), GREEN, 0.7, 2)
    return img


def compose(res: PipelineResult, skin: SkinModel, move_zones=(0.33, 0.67), scale: float = 1.5) -> np.ndarray:
    annotated = draw_annotations(res, skin, move_zones)
    shape = annotated.shape
    fg = _gray3(res.foreground, shape)
    sk = _gray3(res.skin, shape)
    hand_view = _gray3(res.candidate, shape) // 3
    if res.hand is not None:
        hand_view[res.hand.mask > 0] = GREEN
    for img, title in ((annotated, "camera"), (fg, "foreground"), (sk, "skin"), (hand_view, "hand")):
        _put(img, title, (8, shape[0] - 8) if img is not annotated else (shape[1] - 70, 18), CYAN, 0.45)

    grid = np.vstack([np.hstack([annotated, fg]), np.hstack([sk, hand_view])])

    panel = np.zeros((70, grid.shape[1], 3), np.uint8)
    t = res.timings
    stages = ["preprocess", "background", "face", "skin", "hand", "tracking", "fingers"]
    line1 = "  ".join(f"{s}:{t[s]:.1f}" for s in stages if s in t)
    _put(panel, f"FPS {res.fps:4.1f}   total {t.get('total', 0):.1f} ms   phase: {res.phase.value}", (8, 18))
    _put(panel, f"ms  {line1}", (8, 38), (200, 200, 200), 0.4)
    act = res.actions
    events = [n for n in ("punch", "kick", "special", "dodge", "block") if getattr(act, n)]
    _put(panel, f"raw: {res.raw_gesture.value:6s} stable: {res.gesture.value:6s} "
                f"move: {act.move_x:+d}  actions: {','.join(events) or '-'}  "
                f"dodge score: {res.dodge_score:.2f}   Cr{skin.cr_range} Cb{skin.cb_range}",
         (8, 58), (200, 200, 200), 0.4)
    out = np.vstack([grid, panel])
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return out


HELP_TEXT = "keys: [b] relearn background  [c] recalibrate skin  [n] skip calibration  [q] quit"
