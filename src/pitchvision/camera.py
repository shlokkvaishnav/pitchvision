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


def camera_centre(ball_x: np.ndarray, footprint: Footprint, pitch_length: float = PITCH_LENGTH) -> np.ndarray:
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
    limit = max(pitch_length / 2 - half, 0.0)
    return np.clip(centre, -limit, limit)


def visible_mask(
    positions: np.ndarray,
    ball_x: np.ndarray,
    footprint: Footprint,
    pitch: tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
    centre: np.ndarray | None = None,
) -> np.ndarray:
    """Return a (T, N) bool array. True means the camera shows the player.

    positions: (T, N, 2) array of player positions. NaN means no data, and gives False.
    ball_x: (T,) ball position along x.
    centre: (T,) camera centre. Give it to use a centre that you computed on more frames.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (T, N, 2)")
    length, width = pitch
    if centre is None:
        centre = camera_centre(ball_x, footprint, length)
    centre = np.asarray(centre, dtype=float)[:, None]
    x, y = positions[..., 0], positions[..., 1]
    depth = np.clip((y + width / 2) / width, 0.0, 1.0)  # 0 near touchline, 1 far touchline
    half_width = (footprint.near_width + depth * (footprint.far_width - footprint.near_width)) / 2
    inside = np.abs(x - centre) <= half_width
    return inside & np.isfinite(x) & np.isfinite(y)


def polygon_mask(positions: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return a (T, N) bool array. True means the player is inside the real camera footprint.

    corners: (T, 4, 2) convex footprint, counter-clockwise. A frame with NaN corners gives False.
    """
    positions = np.asarray(positions, dtype=float)
    corners = np.asarray(corners, dtype=float)
    a = corners[:, None, :, :]  # (T, 1, 4, 2)
    b = np.roll(corners, -1, axis=1)[:, None, :, :]
    p = positions[:, :, None, :]  # (T, N, 1, 2)
    cross = (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (p[..., 0] - a[..., 0])
    with np.errstate(invalid="ignore"):
        inside = np.all(cross >= 0, axis=-1)
    valid = np.isfinite(positions).all(-1) & np.isfinite(corners).all((-1, -2))[:, None]
    return inside & valid
