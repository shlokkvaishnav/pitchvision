"""Spec 09 (exploratory): the spec 05 model and the Kalman smoother under changed camera masks.

Usage: PYTHONPATH=src python scripts/camera_stress.py [--raw data/raw/skillcorner] [--only zoom_0.7,none]
Needs data/models/05-more-data.pt. Writes results/09-camera-stress.json. CPU only.
Data: the spec 05 test matches. The locked holdout matches are never loaded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from pitchvision import baselines as bl
from pitchvision.anchor import imputation_anchor
from pitchvision.camera import polygon_mask
from pitchvision.metrics import at_cells, cell_scores
from pitchvision.model import AnchoredModel, predict_windows
from pitchvision.prep import FPS, idsse_match, replay_corners, skillcorner_match
from pitchvision.splits import assert_not_holdout
from pitchvision.windows import cut

ROOT = Path(__file__).resolve().parents[1]
MATCHES = {"J03WQQ": "1959846", "J03WR9": "1986691"}  # spec 05 test matches and their donors
STD_SCALE = 1.1747  # spec 05
CUT_EVERY_S = 60.0  # mean time between replay cut starts
BINS_S = [0, 1, 3, 6, 9.6, np.inf]
BIN_NAMES = ["<1s", "1-3s", "3-6s", "6-9.6s", ">=9.6s"]
ROLES = ("goalkeeper", "defender", "midfielder", "forward", "unknown")
CONDITIONS = {
    "none": ("none", 0.0),
    "zoom_0.7": ("zoom", 0.7), "zoom_0.85": ("zoom", 0.85), "zoom_1.15": ("zoom", 1.15), "zoom_1.3": ("zoom", 1.3),
    "lag_0.5s": ("lag", 0.5), "lag_1s": ("lag", 1.0), "lag_2s": ("lag", 2.0),
    "cut_2s": ("cut", 2.0), "cut_4s": ("cut", 4.0), "cut_8s": ("cut", 8.0), "cut_4s_ball": ("cut_ball", 4.0),
}  # fmt: skip


def roles(match: str, player_ids: list[str]) -> np.ndarray:
    """Coarse role of each player column, from kloppy's starting position."""
    from kloppy import sportec
    from kloppy.domain import PositionType as P

    ds = sportec.load_open_tracking_data(match, coordinates="sportec", limit=1)
    pos = {str(p.player_id): p.starting_position for t in ds.metadata.teams for p in t.players}

    def coarse(p) -> str:
        if p is None or p == P.Unknown:
            return "unknown"
        for name, group in (("goalkeeper", P.Goalkeeper), ("defender", P.Defender), ("midfielder", P.Midfielder),
                            ("forward", P.Attacker)):  # fmt: skip
            if p == group or p.is_subtype_of(group):
                return name
        return "unknown"

    return np.array([coarse(pos.get(i)) for i in player_ids])


def zoom(corners: np.ndarray, factor: float) -> np.ndarray:
    centre = corners.mean(axis=1, keepdims=True)
    return centre + factor * (corners - centre)


def lag(corners: np.ndarray, period: np.ndarray, seconds: float) -> np.ndarray:
    k = round(seconds * FPS)
    out = corners.copy()
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        src = np.maximum(rows - k, rows[0])
        out[rows] = corners[src]
    return out


def cut_frames(period: np.ndarray, seconds: float, seed: int) -> np.ndarray:
    """(T,) bool: frames inside a replay cut. Starts: Poisson process, the same for every cut length (fixed seed)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(period), bool)
    n = round(seconds * FPS)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        t = rows[0] + rng.exponential(CUT_EVERY_S * FPS)
        while t < rows[-1]:
            s = int(t)
            out[s : min(s + n, rows[-1] + 1)] = True
            t += rng.exponential(CUT_EVERY_S * FPS)
    return out


def masks(tr, corners: np.ndarray, kind: str, value: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Player visibility (T, N) and ball visibility (T,) for one condition."""
    ball = np.ones(len(tr.period), bool)
    if kind == "zoom":
        corners = zoom(corners, value)
    elif kind == "lag":
        corners = lag(corners, tr.period, value)
    vis = polygon_mask(tr.players, corners)
    if kind in ("cut", "cut_ball"):
        c = cut_frames(tr.period, value, seed)
        vis[c] = False
        if kind == "cut_ball":
            ball = ~c
    return vis, ball


@torch.no_grad()
def score(model, tr, vis: np.ndarray, ball_vis: np.ndarray, role: np.ndarray) -> dict[str, np.ndarray]:
    """Per hidden player cell: errors, coverage, gap length and role."""
    w = cut(tr, vis, length=90, stride=20)
    frames = w.start[:, None] + np.arange(w.pos.shape[1])[None]
    w.visible[..., 22] = ball_vis[frames]
    n = tr.players.shape[1]
    pos = np.concatenate([tr.players, tr.ball[:, None]], axis=1)
    k = bl.kalman(pos, np.concatenate([vis, ball_vis[:, None]], axis=1), FPS)
    w.imp_mean, w.imp_ls = imputation_anchor(k, w, n)
    kal = k.smooth_x[:, :n, :, 0]
    del k
    hidden = ~w.visible[..., :22]
    mu, ls = predict_windows(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, "cpu")
    err, inside = cell_scores(mu, ls + np.log(STD_SCALE), w, hidden)
    _, inside_kal = cell_scores(w.imp_mean, w.imp_ls, w, hidden)
    gl = np.zeros(vis.shape, int)
    for p in np.unique(tr.period):
        rows = tr.period == p
        gl[rows] = bl.gap_lengths(~vis[rows])
    cols = np.broadcast_to(w.cols[:, None, :], hidden.shape)[hidden]
    f = np.broadcast_to(frames[:, :, None], hidden.shape)[hidden]
    return {
        "model": err, "kalman": at_cells(kal, tr, w, hidden),
        "spline": at_cells(bl.by_period(bl.cubic_spline, tr.players, vis, tr.period), tr, w, hidden),
        "inside": inside.astype(bool), "inside_kal": inside_kal.astype(bool),
        "gap_s": gl[f, cols] / FPS, "role": role[cols],
    }  # fmt: skip


def summary(c: dict[str, np.ndarray], sel: np.ndarray) -> dict:
    keep = sel & np.isfinite(c["kalman"]) & np.isfinite(c["model"])
    if not keep.any():
        return {"n_cells": 0}
    out = {"n_cells": int(keep.sum())}
    for m in ("model", "kalman", "spline"):
        ok = keep & np.isfinite(c[m])
        out[f"{m}_median_m"] = float(np.median(c[m][ok]))
        out[f"{m}_mean_m"] = float(c[m][ok].mean())
    out["model_over_kalman_median"] = out["model_median_m"] / out["kalman_median_m"]
    out["coverage_90_model"] = float(c["inside"][keep].mean())
    out["coverage_90_kalman"] = float(c["inside_kal"][keep].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/skillcorner")
    ap.add_argument("--only", default="", help="comma-separated conditions (default: all)")
    args = ap.parse_args()
    assert_not_holdout([f"idsse-{m}" for m in MATCHES])
    names = args.only.split(",") if args.only else list(CONDITIONS)
    model = AnchoredModel(length=90, dim=128, heads=4, depth=4)
    model.load_state_dict(torch.load(ROOT / "data" / "models" / "05-more-data.pt", map_location="cpu"))
    parts: dict[str, list[dict]] = {c: [] for c in names}
    hidden_share: dict[str, list[tuple[int, int]]] = {c: [] for c in names}
    for i, (match, donor) in enumerate(MATCHES.items()):
        tr = idsse_match(match)
        corners = replay_corners(tr, skillcorner_match(donor, Path(args.raw)))
        role = roles(match, tr.player_ids)
        present = np.isfinite(tr.players).all(-1)
        for name in names:
            t0 = time.time()
            kind, value = CONDITIONS[name]
            vis, ball = masks(tr, corners, kind, value, seed=i)
            parts[name].append(score(model, tr, vis, ball, role))
            hidden_share[name].append((int((~vis & present).sum()), int(present.sum())))
            print(f"{match} {name}: {time.time() - t0:.0f} s", flush=True)
    res = {}
    for name in names:
        c = {k: np.concatenate([p[k] for p in parts[name]]) for k in parts[name][0]}
        everything = np.ones(len(c["model"]), bool)
        bins = np.digitize(c["gap_s"], BINS_S[1:-1])
        h = np.array(hidden_share[name]).sum(0)
        res[name] = {
            "hidden_share": float(h[0] / h[1]),
            "pooled": summary(c, everything),
            "by_gap": {b: summary(c, bins == j) for j, b in enumerate(BIN_NAMES)},
            "by_role": {r: summary(c, c["role"] == r) for r in ROLES},
        }
        res[name]["by_gap_cell_share"] = {b: float((bins == j).mean()) for j, b in enumerate(BIN_NAMES)}
        res[name]["by_role_cell_share"] = {r: float((c["role"] == r).mean()) for r in ROLES}
    out = {
        "spec": "09-camera-stress (exploratory, no rule)",
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "matches": [f"idsse-{m}" for m in MATCHES],
        "std_scale": STD_SCALE,
        "cut_every_s": CUT_EVERY_S,
        "conditions": {k: {"kind": CONDITIONS[k][0], "value": CONDITIONS[k][1]} for k in names},
        "results": res,
    }
    dest = ROOT / "results" / ("09-camera-stress.json" if not args.only else "09-camera-stress-partial.json")
    dest.write_text(json.dumps(out, indent=2))
    for name in names:
        p = res[name]["pooled"]
        print(f"{name:12s} hidden {res[name]['hidden_share']:.3f}  model {p['model_median_m']:.2f}  "
              f"kalman {p['kalman_median_m']:.2f}  spline {p['spline_median_m']:.2f}  "
              f"cov {p['coverage_90_model']:.3f} / {p['coverage_90_kalman']:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
