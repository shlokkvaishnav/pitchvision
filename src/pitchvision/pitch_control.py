"""Pitch control (Spearman 2018), as implemented in Laurie Shaw's LaurieOnTracking (MIT), for both teams alike.

Reference for site/pitchcontrol.js (the page computes the same thing; tests/js checks they agree).
Differences from Shaw's code: no goalkeeper or offside term (the demo data has neither); home and away use the same
lambda (Shaw's default kappa_def = 1); the result is the home share of the total control at the end, so
home + away = 1 even when the integration stops at 0.99. Positions and velocities in metres and m/s, pitch centre at (0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    reaction_time: float = 0.7
    vmax: float = 5.0
    tti_sigma: float = 0.45
    lam: float = 4.3
    ball_speed: float = 15.0
    dt: float = 0.04
    max_time: float = 10.0
    tol: float = 0.01
    veto: float = 3.0

    @property
    def time_to_control(self) -> float:
        return self.veto * np.log(10) * (np.sqrt(3) * self.tti_sigma / np.pi + 1 / self.lam)


def grid(nx: int = 32, length: float = 105.0, width: float = 68.0) -> np.ndarray:
    """Cell centres, (ny, nx, 2), ny = round(nx * width / length)."""
    ny = round(nx * width / length)
    xs = (np.arange(nx) + 0.5) * length / nx - length / 2
    ys = (np.arange(ny) + 0.5) * width / ny - width / 2
    return np.stack(np.meshgrid(xs, ys), -1)


def home_control(
    cell: np.ndarray, pos: np.ndarray, vel: np.ndarray, team: np.ndarray, ball: np.ndarray, p: Params = Params()
) -> float:
    """Probability that the home team (team == 0) controls a ball sent to `cell`. pos, vel: (n, 2); team: (n,)."""
    ball_t = float(np.linalg.norm(cell - ball)) / p.ball_speed if np.all(np.isfinite(ball)) else 0.0
    tau = p.reaction_time + np.linalg.norm(cell - (pos + vel * p.reaction_time), axis=1) / p.vmax
    home, away = team == 0, team == 1
    if not away.any() or not home.any():  # a team with no players on the pitch controls nothing
        return float(home.any())
    th, ta = tau[home].min(), tau[away].min()
    if th - max(ball_t, ta) >= p.time_to_control:
        return 0.0
    if ta - max(ball_t, th) >= p.time_to_control:
        return 1.0
    keep = (home & (tau - th < p.time_to_control)) | (away & (tau - ta < p.time_to_control))
    tau, side = tau[keep], team[keep]
    ppcf = np.zeros(len(tau))
    k = np.pi / np.sqrt(3) / p.tti_sigma
    for t in np.arange(ball_t - p.dt, ball_t + p.max_time, p.dt):
        total = ppcf.sum()
        if total >= 1 - p.tol:
            break
        f = 1 / (1 + np.exp(-k * (t - tau)))
        ppcf += (1 - total) * f * p.lam * p.dt  # Euler step of Spearman's equation, all players from the same total
    return float(ppcf[side == 0].sum() / max(ppcf.sum(), 1e-12))


def control_map(
    pos: np.ndarray, vel: np.ndarray, team: np.ndarray, ball: np.ndarray, nx: int = 32, p: Params = Params()
) -> np.ndarray:
    """Home-team control probability on the grid, (ny, nx)."""
    g = grid(nx)
    return np.array([[home_control(c, pos, vel, team, ball, p) for c in row] for row in g])
