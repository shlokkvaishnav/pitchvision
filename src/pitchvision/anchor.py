"""Kalman anchor for masked model version B (specs/04-anchored-model.md, section 2)."""

from __future__ import annotations

import numpy as np

from .baselines import kalman
from .tracking import Tracking
from .windows import HALF_LENGTH, HALF_WIDTH, Windows

SCALE = np.array([HALF_LENGTH, HALF_WIDTH])
MAX_SPEED = 8.0
MIN_STD_M = 0.05
CLIP = 1.05  # spec 04 A1: the pitch plus 5%


def match_kalman(tr: Tracking, visible: np.ndarray, fps: float):
    """Kalman on the players and the ball (the 23rd track, always visible). Positions in metres."""
    pos = np.concatenate([tr.players, tr.ball[:, None]], axis=1)
    vis = np.concatenate([visible, np.ones((len(visible), 1), bool)], axis=1)
    return kalman(pos, vis, fps)


def _gather(arr: np.ndarray, w: Windows, n_players: int) -> np.ndarray:
    """Take (T, N+1, ...) match arrays into (B, L, 23, ...) window arrays."""
    length = w.pos.shape[1]
    frames = w.start[:, None] + np.arange(length)[None]  # (B, L)
    cols = np.concatenate([w.cols, np.full((len(w.cols), 1), n_players)], axis=1)  # (B, 23)
    return arr[frames[:, :, None], cols[:, None, :]]


def _normalise(mean_m: np.ndarray, var_m: np.ndarray):
    std = np.sqrt(np.maximum(var_m, MIN_STD_M**2))
    return np.clip(mean_m / SCALE, -CLIP, CLIP), np.log(std / SCALE)


def imputation_anchor(k, w: Windows, n_players: int):
    """Smoother mean and log-std, normalised. Shapes (B, L, 23, 2)."""
    mean = _gather(k.smooth_x[..., 0], w, n_players)
    var = _gather(k.smooth_p[..., 0, 0], w, n_players)
    return _normalise(mean, var)


def forecast_anchor(k, w: Windows, n_players: int, horizon: int):
    """Causal anchor: filter for the observed frames, then a prediction from the last observed frame."""
    length = w.pos.shape[1]
    split = length - horizon
    mean = _gather(k.filt_x[..., 0], w, n_players).copy()
    var = _gather(k.filt_p[..., 0, 0], w, n_players).copy()
    x = _gather(k.filt_x, w, n_players)[:, split - 1].copy()  # (B, 23, 2 axes, 2)
    p = _gather(k.filt_p, w, n_players)[:, split - 1].copy()
    speed = np.linalg.norm(x[..., 1], axis=-1, keepdims=True)
    x[..., 1] *= np.minimum(1.0, MAX_SPEED / np.maximum(speed, 1e-9))
    for h in range(horizon):
        x = np.einsum("ij,bnaj->bnai", k.f, x)
        p = np.einsum("ij,bnajk,lk->bnail", k.f, p, k.f) + k.q
        mean[:, split + h] = x[..., 0]
        var[:, split + h] = p[..., 0, 0]
    return _normalise(mean, var)
