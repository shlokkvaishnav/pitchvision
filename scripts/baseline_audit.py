"""Spec 06: baseline audit on all 8 development matches (no training).

Usage: PYTHONPATH=src python scripts/baseline_audit.py --out results [--raw data/raw/skillcorner] [--only metrica-1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_more as tm

from pitchvision import baselines as bl
from pitchvision.anchor import SCALE
from pitchvision.model import CHI2_2DOF_90
from pitchvision.prep import FPS

MATCHES = ["metrica-1", "metrica-2", "metrica-3", *[f"idsse-{m}" for m in tm.IDSSE_DEV]]
BINS_S = [0, 1, 3, 6, 9.6, np.inf]
BIN_NAMES = ["<1s", "1-3s", "3-6s", "6-9.6s", ">=9.6s"]
IMPUTERS = ("linear", "spline", "kalman")
FORECASTERS = ("cv_last", "cv_smooth", "stationary", "kalman_filter")


def imputation_parts(tr, vis) -> dict[str, np.ndarray]:
    hidden = ~vis & np.isfinite(tr.players).all(-1)
    gl = np.zeros(hidden.shape, int)
    est = {"linear": bl.by_period(bl.linear_interpolation, tr.players, vis, tr.period),
           "spline": bl.by_period(bl.cubic_spline, tr.players, vis, tr.period)}  # fmt: skip
    kal_mean = np.full(tr.players.shape, np.nan)
    kal_std = np.full(tr.players.shape, np.nan)
    for p in np.unique(tr.period):
        rows = tr.period == p
        gl[rows] = bl.gap_lengths(~vis[rows])
        k = bl.kalman(tr.players[rows], vis[rows], FPS)
        kal_mean[rows] = k.smooth_x[..., 0]
        kal_std[rows] = np.sqrt(np.maximum(k.smooth_p[..., 0, 0], 1e-6))
    est["kalman"] = kal_mean
    out = {"gap_s": gl[hidden] / FPS}
    for name, e in est.items():
        out[name] = np.linalg.norm(e[hidden] - tr.players[hidden], axis=-1)
    z2 = (((kal_mean - tr.players) / kal_std) ** 2).sum(-1)
    out["kalman_inside"] = (z2[hidden] <= CHI2_2DOF_90).astype(float)
    return out


def forecast_parts(w) -> dict[str, np.ndarray]:
    obs = w.visible.copy()
    obs[:, -tm.HORIZON :, :] = False
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


def summarise(imp: list[dict], fc: list[dict]) -> dict:
    c = {k: np.concatenate([p[k] for p in imp]) for k in imp[0]}
    bins = np.digitize(c["gap_s"], BINS_S[1:-1])
    res = {"n_hidden_cells": int(c["gap_s"].size), "imputation": {}, "bin_fraction": {}}
    for b, name in enumerate(BIN_NAMES):
        res["bin_fraction"][name] = float((bins == b).mean())
    for m in IMPUTERS:
        e = c[m]
        ok = np.isfinite(e)
        row = {"median_m": float(np.median(e[ok])), "mean_m": float(e[ok].mean())}
        for b, name in enumerate(BIN_NAMES):
            sel = ok & (bins == b)
            row[name] = {"median_m": float(np.median(e[sel])), "mean_m": float(e[sel].mean())} if sel.any() else None
        res["imputation"][m] = row
    res["kalman_coverage_90"] = float(c["kalman_inside"].mean())
    res["kalman_coverage_90_by_bin"] = {
        name: float(c["kalman_inside"][bins == b].mean()) if (bins == b).any() else None
        for b, name in enumerate(BIN_NAMES)
    }
    f = {k: np.concatenate([p[k] for p in fc]) for k in fc[0]}
    same = np.all([np.isfinite(f[k]) for k in FORECASTERS], axis=0)
    res["forecast_4s_mean_m"] = {k: float(f[k][same].mean()) for k in FORECASTERS}
    res["forecast_cell_fraction"] = float(same.mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--raw", default="data/raw/skillcorner")
    ap.add_argument("--only", nargs="*", default=None, help="smoke test only: a subset of matches")
    ap.add_argument("--commit", default=os.environ.get("PITCHVISION_COMMIT", "unknown"))
    args = ap.parse_args()
    dn = tm.donors()
    per, imp_all, fc_all = {}, [], []
    for m in args.only or MATCHES:
        tr, vis, w = tm.load_match(m, dn[m], Path(args.raw))
        imp, fc = imputation_parts(tr, vis), forecast_parts(w)
        per[m] = summarise([imp], [fc])
        imp_all.append(imp)
        fc_all.append(fc)
        print(m, json.dumps({k: per[m][k] for k in ("imputation", "forecast_4s_mean_m")}), flush=True)
    pooled = summarise(imp_all, fc_all)
    bars = {
        "imputation_bar": min(IMPUTERS, key=lambda k: pooled["imputation"][k]["median_m"]),
        "forecast_bar": min(FORECASTERS, key=lambda k: pooled["forecast_4s_mean_m"][k]),
    }
    out = {"spec": "06-baseline-audit", "commit": args.commit, "pooled": pooled, "bars": bars, "per_match": per}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "06-baseline-audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"pooled": pooled, "bars": bars}, indent=2))


if __name__ == "__main__":
    main()
