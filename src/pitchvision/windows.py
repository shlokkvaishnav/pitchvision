"""Cut a match into fixed windows of 22 players + ball (see specs/03-masked-model.md, section 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import Tracking

HALF_LENGTH, HALF_WIDTH = 52.5, 34.0
HOME, AWAY, BALL = 0, 1, 2


@dataclass
class Windows:
    pos: np.ndarray  # (B, L, 23, 2) normalised positions, NaN = no truth
    visible: np.ndarray  # (B, L, 23) camera-visible; the ball is always True
    agent_type: np.ndarray  # (23,) HOME, AWAY or BALL
    start: np.ndarray  # (B,) first frame of each window in the match


def cut(tr: Tracking, visible: np.ndarray, length: int = 90, stride: int = 20) -> Windows:
    """Return the windows with exactly 11 players of each team present in all frames, in one period."""
    teams = np.asarray(tr.teams)
    present = np.isfinite(tr.players).all(-1)  # (T, N)
    pos, vis, starts = [], [], []
    t = len(tr.period)
    for s in range(0, t - length + 1, stride):
        rows = slice(s, s + length)
        if tr.period[s] != tr.period[s + length - 1] or not np.isfinite(tr.ball[rows]).all():
            continue
        full = present[rows].all(0)
        home = np.flatnonzero(full & (teams == "home"))
        away = np.flatnonzero(full & (teams == "away"))
        if home.size != 11 or away.size != 11:
            continue
        cols = np.concatenate([home, away])
        p = np.concatenate([tr.players[rows][:, cols], tr.ball[rows][:, None]], axis=1)
        v = np.concatenate([visible[rows][:, cols], np.ones((length, 1), dtype=bool)], axis=1)
        pos.append(p)
        vis.append(v)
        starts.append(s)
    if not pos:
        return Windows(np.zeros((0, length, 23, 2)), np.zeros((0, length, 23), bool), _types(), np.zeros(0, int))
    out = np.stack(pos)
    out[..., 0] /= HALF_LENGTH
    out[..., 1] /= HALF_WIDTH
    return Windows(out, np.stack(vis), _types(), np.asarray(starts))


def _types() -> np.ndarray:
    return np.array([HOME] * 11 + [AWAY] * 11 + [BALL])


def task_mask(visible: np.ndarray, forecast: np.ndarray, horizon: int = 40) -> np.ndarray:
    """Observed cells: camera-visible, and for forecast windows not in the last `horizon` frames.

    visible: (B, L, A). forecast: (B,) bool.
    """
    obs = visible.copy()
    obs[forecast, -horizon:, :] = False
    return obs
