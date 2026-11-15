"""Imputation baselines for hidden players (spec 00, rule R1).

All functions take the true positions and a visibility mask, and return a full (T, N, 2) estimate.
They use only the visible positions. Imputation is offline: a baseline can use frames after the gap.
"""

from __future__ import annotations

import warnings

import numpy as np


def _observed(positions: np.ndarray, visible: np.ndarray) -> np.ndarray:
    obs = np.array(positions, dtype=float, copy=True)
    obs[~visible] = np.nan
    return obs


def hold_last(positions: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Keep the last visible position. Before the first visible frame, use the first visible position."""
    obs = _observed(positions, visible)
    t, n, _ = obs.shape
    out = np.full_like(obs, np.nan)
    for j in range(n):
        seen = np.flatnonzero(visible[:, j])
        if seen.size == 0:
            continue
        idx = np.maximum.accumulate(np.where(visible[:, j], np.arange(t), -1))
        idx[idx < 0] = seen[0]
        out[:, j] = obs[idx, j]
    return out


def constant_velocity(positions: np.ndarray, visible: np.ndarray, fps: float, max_speed: float = 8.0) -> np.ndarray:
    """Move from the last visible position with the last visible velocity (causal).

    The velocity comes from the last two visible frames of the same run, and its speed is capped at max_speed (m/s).
    """
    obs = _observed(positions, visible)
    out = hold_last(positions, visible)
    t, n, _ = obs.shape
    dt = 1.0 / fps
    for j in range(n):
        v = np.zeros(2)
        last = None
        for k in range(t):
            if visible[k, j]:
                if last is not None and last == k - 1:
                    v = (obs[k, j] - obs[k - 1, j]) / dt
                    speed = np.linalg.norm(v)
                    if speed > max_speed:
                        v *= max_speed / speed
                elif last is None or last < k - 1:
                    v = np.zeros(2)
                last = k
            elif last is not None:
                out[k, j] = obs[last, j] + v * (k - last) * dt
    return out


def linear_interpolation(positions: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Straight line between the last visible position before the gap and the first one after it.

    At the start or the end of the match, hold the nearest visible position.
    """
    obs = _observed(positions, visible)
    t, n, _ = obs.shape
    out = np.full_like(obs, np.nan)
    frames = np.arange(t)
    for j in range(n):
        seen = np.flatnonzero(visible[:, j])
        if seen.size == 0:
            continue
        for d in range(2):
            out[:, j, d] = np.interp(frames, seen, obs[seen, j, d])
    return out


def formation_template(
    positions: np.ndarray, visible: np.ndarray, teams: list[str], offsets: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Centroid of the other visible team-mates, plus the mean offset of the player from that centroid.

    offsets: (N, 2) offsets from training data. If None, they are computed here from the visible frames.
    Returns (estimate, offsets).
    """
    obs = _observed(positions, visible)
    teams_arr = np.asarray(teams)
    t, n, _ = obs.shape
    # Leave-one-out centroid: for each player, the mean of the OTHER visible team-mates.
    # Thus the centroid does not change when the player itself is hidden.
    filled = np.nan_to_num(obs)
    seen = visible & np.isfinite(obs).all(-1)
    centroid = np.full((t, n, 2), np.nan)
    for team in np.unique(teams_arr):
        cols = np.flatnonzero(teams_arr == team)
        total = (filled[:, cols] * seen[:, cols, None]).sum(1)  # (T, 2)
        count = seen[:, cols].sum(1)  # (T,)
        others_sum = total[:, None, :] - filled[:, cols] * seen[:, cols, None]
        others_n = count[:, None] - seen[:, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            centroid[:, cols] = np.where(others_n[..., None] > 0, others_sum / others_n[..., None], np.nan)
    if offsets is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a player with no visible frame gives a NaN mean
            offsets = np.nanmean(obs - centroid, axis=0)
        offsets = np.nan_to_num(offsets)
    est = centroid + offsets[None]
    # frames with no visible team-mate: hold the last estimate
    fallback = hold_last(np.nan_to_num(est), np.isfinite(est).all(-1))
    est = np.where(np.isfinite(est), est, fallback)
    return est, offsets


def kalman_smoother(
    positions: np.ndarray, visible: np.ndarray, fps: float, accel_std: float = 3.0, obs_std: float = 0.3
) -> np.ndarray:
    """Rauch-Tung-Striebel smoother with a constant-velocity model, run on all players at once."""
    obs = _observed(positions, visible)
    t, n, _ = obs.shape
    dt = 1.0 / fps
    f = np.array([[1, dt], [0, 1]])
    q = accel_std**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    r = obs_std**2
    # state per player and axis: [position, velocity]; the two axes are independent
    xs = np.zeros((t, n, 2, 2))
    ps = np.zeros((t, n, 2, 2, 2))
    xp_all = np.zeros_like(xs)
    pp_all = np.zeros_like(ps)
    first = np.where(visible.any(0), visible.argmax(0), 0)
    x = np.zeros((n, 2, 2))
    x[..., 0] = np.nan_to_num(obs[first, np.arange(n)])
    p = np.broadcast_to(np.diag([25.0, 25.0]), (n, 2, 2, 2)).copy()
    for k in range(t):
        xp = np.einsum("ij,naj->nai", f, x)
        pp = np.einsum("ij,najk,lk->nail", f, p, f) + q
        xp_all[k], pp_all[k] = xp, pp
        z = obs[k]  # (n, 2)
        has = visible[k][:, None] & np.isfinite(z)
        s = pp[..., 0, 0] + r
        gain = pp[..., :, 0] / s[..., None]  # (n, 2axes, 2)
        innov = np.where(has, np.nan_to_num(z) - xp[..., 0], 0.0)
        x = xp + gain * innov[..., None]
        update = gain[..., :, None] * pp[..., 0, None, :]
        p = pp - np.where(has[..., None, None], update, 0.0)
        xs[k], ps[k] = x, p
    out = xs.copy()
    for k in range(t - 2, -1, -1):
        c = np.einsum("naij,kj,nakl->nail", ps[k], f, np.linalg.inv(pp_all[k + 1]))
        out[k] = xs[k] + np.einsum("naij,naj->nai", c, out[k + 1] - xp_all[k + 1])
    return out[..., 0]


def errors(estimate: np.ndarray, positions: np.ndarray, hidden: np.ndarray) -> dict[str, float]:
    """Position error in metres on the hidden player-frames that have a true position."""
    keep = hidden & np.isfinite(positions).all(-1) & np.isfinite(estimate).all(-1)
    e = np.linalg.norm(estimate[keep] - positions[keep], axis=-1)
    return {"median": float(np.median(e)), "mean": float(e.mean()), "p90": float(np.quantile(e, 0.9)), "n": int(e.size)}


def gap_lengths(hidden: np.ndarray) -> np.ndarray:
    """For each hidden player-frame, the length (in frames) of the hidden run that contains it. 0 if visible."""
    t, n = hidden.shape
    out = np.zeros((t, n), dtype=int)
    for j in range(n):
        h = hidden[:, j].astype(int)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], h, [0]])))
        for start, stop in zip(edges[::2], edges[1::2]):
            out[start:stop, j] = stop - start
    return out
