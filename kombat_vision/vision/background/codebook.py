"""Codebook background subtraction (Kim, Chalidabhongse, Harwood, Davis - 2005).

Each pixel keeps a small "codebook" of codewords. A codeword stores:
    v      mean color (R, G, B)
    Imin   smallest brightness seen
    Imax   largest brightness seen
    f      how many times it matched
    lambda maximum negative run-length (longest period it was NOT seen)
    p, q   first and last access time

A new pixel x matches a codeword when:
    colordist(x, v) <= epsilon                  (same chromaticity)
    alpha * Imax <= |x| <= min(beta * Imax, Imin / alpha)   (plausible brightness)

Training adds/updates codewords. At the end of training, codewords with
lambda > N/2 are discarded (they were absent most of the time: moving objects).
Detection: a pixel is background if it matches any remaining codeword.

This version is vectorized with NumPy and keeps a fixed number of codeword
slots per pixel (max_codewords), which is fast enough for real time at 320x240.
"""

from __future__ import annotations

import numpy as np

from kombat_vision.config import CodebookConfig
from kombat_vision.vision.background.base import BackgroundModel


class CodebookBackground(BackgroundModel):
    def __init__(self, cfg: CodebookConfig, warmup_frames: int):
        super().__init__(warmup_frames)
        self.cfg = cfg
        self._shape: tuple[int, int] | None = None
        self.active: np.ndarray | None = None

    # ------------------------------------------------------------------ setup
    def _allocate(self, h: int, w: int) -> None:
        k = self.cfg.max_codewords
        self._shape = (h, w)
        self.v = np.zeros((h, w, k, 3), np.float32)
        self.imin = np.zeros((h, w, k), np.float32)
        self.imax = np.zeros((h, w, k), np.float32)
        self.f = np.zeros((h, w, k), np.int32)
        self.lam = np.zeros((h, w, k), np.int32)
        self.p = np.zeros((h, w, k), np.int32)
        self.q = np.zeros((h, w, k), np.int32)
        self.used = np.zeros((h, w, k), bool)
        self.active = None

    def reset(self) -> None:
        super().reset()
        self._shape = None
        self.active = None

    # --------------------------------------------------------------- matching
    def _matches(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (matches[h,w,k], brightness[h,w]) for frame x (float32, h x w x 3)."""
        eps = 1e-6
        x2 = np.einsum("hwc,hwc->hw", x, x)                       # |x|^2
        brightness = np.sqrt(x2)
        dot = np.einsum("hwc,hwkc->hwk", x, self.v)               # <x, v>
        v2 = np.einsum("hwkc,hwkc->hwk", self.v, self.v) + eps    # |v|^2
        colordist2 = np.maximum(x2[..., None] - dot * dot / v2, 0.0)
        color_ok = colordist2 <= self.cfg.color_epsilon ** 2

        b = brightness[..., None]
        i_low = self.cfg.alpha * self.imax
        i_high = np.minimum(self.cfg.beta * self.imax, self.imin / self.cfg.alpha)
        bright_ok = (b >= i_low) & (b <= i_high)
        return self.used & color_ok & bright_ok, brightness

    # --------------------------------------------------------------- training
    def _train(self, x: np.ndarray, t: int) -> None:
        matches, brightness = self._matches(x)
        has_match = matches.any(axis=-1)

        # 1) Update the first matching codeword of each matched pixel.
        rows, cols = np.nonzero(has_match)
        if rows.size:
            k = np.argmax(matches[rows, cols], axis=-1)
            f = self.f[rows, cols, k][:, None].astype(np.float32)
            self.v[rows, cols, k] = (f * self.v[rows, cols, k] + x[rows, cols]) / (f + 1)
            br = brightness[rows, cols]
            self.imin[rows, cols, k] = np.minimum(self.imin[rows, cols, k], br)
            self.imax[rows, cols, k] = np.maximum(self.imax[rows, cols, k], br)
            self.f[rows, cols, k] += 1
            self.lam[rows, cols, k] = np.maximum(self.lam[rows, cols, k], t - self.q[rows, cols, k])
            self.q[rows, cols, k] = t

        # 2) Create a codeword where nothing matched: first free slot, otherwise
        #    replace the stalest codeword (largest lambda).
        rows, cols = np.nonzero(~has_match)
        if rows.size:
            free = ~self.used[rows, cols]
            slot = np.where(free.any(axis=-1), np.argmax(free, axis=-1),
                            np.argmax(self.lam[rows, cols], axis=-1))
            br = brightness[rows, cols]
            self.v[rows, cols, slot] = x[rows, cols]
            self.imin[rows, cols, slot] = br
            self.imax[rows, cols, slot] = br
            self.f[rows, cols, slot] = 1
            self.lam[rows, cols, slot] = t - 1
            self.p[rows, cols, slot] = t
            self.q[rows, cols, slot] = t
            self.used[rows, cols, slot] = True

    def _finish_training(self) -> None:
        n = self.frames_seen
        # Wrap-around: time not seen between the last and the first access.
        wrap = n - self.q + self.p - 1
        self.lam = np.maximum(self.lam, wrap)
        self.active = self.used & (self.lam <= n // 2)
        self.used = self.active

    # -------------------------------------------------------------- interface
    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        x = frame_bgr.astype(np.float32)
        if self._shape != x.shape[:2]:
            self._allocate(*x.shape[:2])

        if not self.ready:
            self.frames_seen += 1
            self._train(x, self.frames_seen)
            if self.ready:
                self._finish_training()
            return np.zeros(x.shape[:2], np.uint8)

        matches, _ = self._matches(x)
        background = matches.any(axis=-1)
        return np.where(background, 0, 255).astype(np.uint8)
