"""Scores on hidden player cells: imputation error, 90% ellipse coverage, and the 4 s forecast error."""

from __future__ import annotations

import numpy as np
import torch

from . import baselines as bl
from .anchor import SCALE
from .dataset import HORIZON
from .model import inside_ellipse, predict_windows
from .prep import FPS

FORECASTERS = ("cv_last", "cv_smooth", "stationary", "kalman_filter")


def cell_scores(mu, ls, w, hidden) -> tuple[np.ndarray, np.ndarray]:
    """Error (m) and 'inside the 90% ellipse' for each hidden player cell of the windows."""
    err = np.linalg.norm((mu[..., :22, :] - w.pos[..., :22, :]) * SCALE, axis=-1)[hidden]
    inside = inside_ellipse(
        torch.as_tensor(mu[..., :22, :]), torch.as_tensor(ls[..., :22, :]), torch.as_tensor(w.pos[..., :22, :])
    ).numpy()[hidden]
    return err, inside


def at_cells(est: np.ndarray, tr, w, hidden: np.ndarray) -> np.ndarray:
    """Error (m) of a full-match estimate (T, N, 2) on the hidden cells of the windows."""
    frames = np.broadcast_to(w.start[:, None, None] + np.arange(w.pos.shape[1])[None, :, None], hidden.shape)
    cols = np.broadcast_to(w.cols[:, None, :], hidden.shape)
    f, c = frames[hidden], cols[hidden]
    return np.linalg.norm(est[f, c] - tr.players[f, c], axis=-1)


def forecast_baselines(w) -> dict[str, np.ndarray]:
    """4 s error of the classical forecasts of spec 06 (last 40 frames hidden), one value per window and player."""
    obs = w.visible.copy()
    obs[:, -HORIZON:, :] = False
    pos_m = w.pos[:, :, :22] * SCALE
    truth = pos_m[:, -1]
    out = {}
    for name, fn in (("cv_last", bl.constant_velocity), ("cv_smooth", bl.constant_velocity_smooth)):
        est = np.stack([fn(pos_m[b], obs[b, :, :22], FPS)[-1] for b in range(len(w.pos))])
        out[name] = np.linalg.norm(est - truth, axis=-1).ravel()
    st = np.stack([bl.hold_last(pos_m[b], obs[b, :, :22])[-1] for b in range(len(w.pos))])
    out["stationary"] = np.linalg.norm(st - truth, axis=-1).ravel()
    out["kalman_filter"] = np.linalg.norm(w.fc_mean[:, -1, :22] * SCALE - truth, axis=-1).ravel()
    return out


def match_cell_errors(model, tr, vis, w, device, log_scale: float = 0.0) -> dict[str, np.ndarray]:
    """Raw per-cell errors of the model, its anchor, the Kalman smoother and constant velocity on one match."""
    hidden = ~w.visible[..., :22]
    mu, ls = predict_windows(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, device)
    ls = ls + log_scale
    err, inside = cell_scores(mu, ls, w, hidden)
    err_anchor, inside_anchor = cell_scores(w.imp_mean, w.imp_ls, w, hidden)
    z2 = (((mu[..., :22, :] - w.pos[..., :22, :]) / np.exp(ls[..., :22, :])) ** 2)[hidden]
    err_kal = at_cells(bl.kalman_smoother(tr.players, vis, FPS), tr, w, hidden)

    obs_f = w.visible.copy()
    obs_f[:, -HORIZON:, :] = False
    mu_f, _ = predict_windows(model, w.pos, obs_f, w.agent_type, w.fc_mean, w.fc_ls, device)
    err_f = np.linalg.norm((mu_f[:, -1, :22] - w.pos[:, -1, :22]) * SCALE, axis=-1).ravel()
    err_fa = np.linalg.norm((w.fc_mean[:, -1, :22] - w.pos[:, -1, :22]) * SCALE, axis=-1).ravel()
    cv = np.stack(
        [bl.constant_velocity(w.pos[b, :, :22] * SCALE, obs_f[b, :, :22], FPS)[-1] for b in range(len(w.pos))]
    )
    err_cv = np.linalg.norm(cv - w.pos[:, -1, :22] * SCALE, axis=-1).ravel()
    return {
        "err": err, "inside": inside, "err_anchor": err_anchor, "inside_anchor": inside_anchor, "z2": z2,
        "err_kal": err_kal, "err_f": err_f, "err_fa": err_fa, "err_cv": err_cv,
    }  # fmt: skip


def pooled(parts: list[dict]) -> dict:
    """Pooled metrics over matches. Imputation on cells where the Kalman smoother has an estimate;
    forecast on cells where constant velocity has an estimate (spec 04 A1)."""
    c = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    keep = np.isfinite(c["err_kal"])
    same = np.isfinite(c["err_cv"])
    return {
        "n_hidden_cells": int(c["err"].size),
        "imputation_median_m": float(np.median(c["err"][keep])),
        "kalman_same_cells_median_m": float(np.median(c["err_kal"][keep])),
        "anchor_imputation_median_m": float(np.median(c["err_anchor"][keep])),
        "coverage_90": float(c["inside"].mean()),
        "anchor_coverage_90": float(c["inside_anchor"].mean()),
        "forecast_4s_mean_m": float(c["err_f"][same].mean()),
        "anchor_forecast_4s_mean_m": float(c["err_fa"][same].mean()),
        "constant_velocity_4s_mean_m": float(c["err_cv"][same].mean()),
        "forecast_4s_mean_all_cells_m": float(c["err_f"].mean()),
        "m3_cell_fraction": float(same.mean()),
    }
