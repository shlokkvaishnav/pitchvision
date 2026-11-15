"""Simulated broadcast camera view (see specs/00-charter.md, section 5).

Coordinates are in metres. The origin is the centre spot.
x runs along the length of the pitch. y runs across the width.
The main camera is on the near touchline (negative y).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


@dataclass(frozen=True)
class Footprint:
    """Trapezoid on the pitch that the camera shows. Spec 01 sets the values."""

    near_width: float  # width along x at the near touchline (m)
    far_width: float  # width along x at the far touchline (m); larger, because of perspective
    pan_smoothing: float = 0.9  # 0 = camera jumps to the ball, near 1 = slow pan

    def __post_init__(self) -> None:
        if not 0 < self.near_width <= self.far_width:
            raise ValueError("need 0 < near_width <= far_width")
        if not 0 <= self.pan_smoothing < 1:
            raise ValueError("need 0 <= pan_smoothing < 1")


def camera_centre(ball_x: np.ndarray, footprint: Footprint) -> np.ndarray:
    """Smooth pan along x that follows the ball. Missing ball positions keep the last centre."""
    ball_x = np.asarray(ball_x, dtype=float)
    centre = np.empty_like(ball_x)
    a = footprint.pan_smoothing
    current = 0.0
    for t, bx in enumerate(ball_x):
        if np.isfinite(bx):
            current = a * current + (1 - a) * bx if t else bx
        centre[t] = current
    half = footprint.far_width / 2
    limit = max(PITCH_LENGTH / 2 - half, 0.0)
    return np.clip(centre, -limit, limit)


def visible_mask(positions: np.ndarray, ball_x: np.ndarray, footprint: Footprint) -> np.ndarray:
    """Return a (T, N) bool array. True means the camera shows the player.

    positions: (T, N, 2) array of player positions. NaN means no data, and gives False.
    ball_x: (T,) ball position along x.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (T, N, 2)")
    centre = camera_centre(ball_x, footprint)[:, None]
    x, y = positions[..., 0], positions[..., 1]
    depth = np.clip((y + PITCH_WIDTH / 2) / PITCH_WIDTH, 0.0, 1.0)  # 0 near touchline, 1 far touchline
    half_width = (footprint.near_width + depth * (footprint.far_width - footprint.near_width)) / 2
    inside = np.abs(x - centre) <= half_width
    return inside & np.isfinite(x) & np.isfinite(y)
