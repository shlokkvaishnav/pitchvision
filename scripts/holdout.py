"""Spec 07: the final test of the spec 05 model on the locked holdout matches.

Usage:
  PYTHONPATH=src python scripts/holdout.py --dry-run     # development match J03WQQ, checks the code, does not count
  PYTHONPATH=src python scripts/holdout.py               # the one holdout run: J03WN1 and J03WOH
Writes results/07-holdout.json (or results/07-dry-run.json). CPU only, no training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_audit as ba
import train_anchored as ta
import train_more as tm

from pitchvision import baselines as bl
from pitchvision.anchor import SCALE, forecast_anchor, imputation_anchor, match_kalman
from pitchvision.model import AnchoredModel
from pitchvision.prep import FPS, idsse_match, replay_mask, skillcorner_match
from pitchvision.windows import cut

ROOT = Path(__file__).resolve().parents[1]
STD_SCALE = 1.1747  # spec 05, fitted on Metrica 3
HOLDOUT = {"J03WN1": "1996435", "J03WOH": "1996436"}  # donors: the 9th and 10th sorted SkillCorner IDs
DRY_RUN = {"J03WQQ": "1959846"}  # the spec 05 donor of this development match
IMPUTERS = ("model", "constant_velocity", "linear", "spline", "kalman", "formation_template")
MAX_SPEED, MAX_ACCEL = 12.0, 10.0


def load(match: str, donor: str, raw: Path, holdout: bool):
    tr = idsse_match(match, allow_holdout=holdout)
    vis = replay_mask(tr, skillcorner_match(donor, raw))
    w = cut(tr, vis, length=90, stride=20)
    k = match_kalman(tr, vis, FPS)
    n = tr.players.shape[1]
    w.imp_mean, w.imp_ls = imputation_anchor(k, w, n)
    w.fc_mean, w.fc_ls = forecast_anchor(k, w, n, tm.HORIZON)
    print(f"idsse-{match}: {w.pos.shape[0]} windows, donor {donor}", flush=True)
    return tr, vis, w


def _at(est: np.ndarray, tr, w, hidden: np.ndarray) -> np.ndarray:
    frames = np.broadcast_to(w.start[:, None, None] + np.arange(w.pos.shape[1])[None, :, None], hidden.shape)
    cols = np.broadcast_to(w.cols[:, None, :], hidden.shape)
    f, c = frames[hidden], cols[hidden]
    return np.linalg.norm(est[f, c] - tr.players[f, c], axis=-1)


def implausible(traj_m: np.ndarray) -> np.ndarray:
    """(B, T, 22, 2) metres -> (B, T-2, 22) flags: speed > 12 m/s or acceleration > 10 m/s² at frames 2..T-1."""
    v = np.linalg.norm(np.diff(traj_m, axis=1), axis=-1) * FPS
    a = np.linalg.norm(np.diff(traj_m, n=2, axis=1), axis=-1) * FPS**2
    return (v[:, 1:] > MAX_SPEED) | (a > MAX_ACCEL)


@torch.no_grad()
def parts(model, tr, vis, w) -> dict[str, np.ndarray]:
    hidden = ~w.visible[..., :22]
    mu, ls = ta.predict(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, "cpu")
    err, inside = ta._cells(mu, ls + np.log(STD_SCALE), w, hidden)
    est = {
        "constant_velocity": bl.by_period(bl.constant_velocity, tr.players, vis, tr.period, fps=FPS),
        "linear": bl.by_period(bl.linear_interpolation, tr.players, vis, tr.period),
        "spline": bl.by_period(bl.cubic_spline, tr.players, vis, tr.period),
        "kalman": bl.by_period(bl.kalman_smoother, tr.players, vis, tr.period, fps=FPS),
        "formation_template": bl.formation_template_by_period(tr.players, vis, tr.teams, tr.period),
    }
    out = {"model": err, "inside": inside.astype(float)}
    out.update({k: _at(e, tr, w, hidden) for k, e in est.items()})

    # forecast: last 40 frames hidden
    obs_f = w.visible.copy()
    obs_f[:, -tm.HORIZON :, :] = False
    mu_f, _ = ta.predict(model, w.pos, obs_f, w.agent_type, w.fc_mean, w.fc_ls, "cpu")
    out.update({f"fc_{k}": v for k, v in ba.forecast_parts(w).items()})
    out["fc_model"] = np.linalg.norm((mu_f[:, -1, :22] - w.pos[:, -1, :22]) * SCALE, axis=-1).ravel()

    # plausible motion over the 40 forecast frames (2 earlier frames for the differences)
    sl = slice(-tm.HORIZON - 2, None)
    truth = w.pos[:, sl, :22] * SCALE
    ok = np.isfinite(truth).all(-1)
    ok = ok[:, 2:] & ok[:, 1:-1] & ok[:, :-2]
    out["bad_model"] = implausible(mu_f[:, sl, :22] * SCALE)[ok].astype(float)
    # report only (amendment A1): a fixed 9-frame moving average over time on the forecast
    sm = uniform_filter1d(mu_f[:, sl, :22] * SCALE, size=9, axis=1, mode="nearest")
    out["bad_model_smooth"] = implausible(sm)[ok].astype(float)
    out["fc_model_smooth"] = np.linalg.norm(sm[:, -1] - w.pos[:, -1, :22] * SCALE, axis=-1).ravel()
    out["bad_truth"] = implausible(np.nan_to_num(truth))[ok].astype(float)
    return out


def summarise(ps: list[dict]) -> dict:
    c = {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}
    same = np.all([np.isfinite(c[k]) for k in IMPUTERS], axis=0)
    baselines = [k for k in IMPUTERS if k != "model"]
    med = {k: float(np.median(c[k][same])) for k in IMPUTERS}
    best = min(baselines, key=med.get)
    fc_names = [f"fc_{k}" for k in ba.FORECASTERS]
    fsame = np.all([np.isfinite(c[k]) for k in [*fc_names, "fc_model"]], axis=0)
    fmean = {k[3:]: float(c[k][fsame].mean()) for k in [*fc_names, "fc_model"]}
    smooth_4s = float(c["fc_model_smooth"][fsame].mean())
    fbest = min(ba.FORECASTERS, key=fmean.get)
    finite_truth = np.isfinite(c["model"])
    res = {
        "n_hidden_cells": int(c["model"].size),
        "imputation_cell_fraction": float(same.mean()),
        "imputation_median_m": med,
        "imputation_mean_m": {k: float(c[k][same].mean()) for k in IMPUTERS},
        "best_imputation_baseline": best,
        "coverage_90": float(c["inside"][finite_truth].mean()),
        "forecast_cell_fraction": float(fsame.mean()),
        "forecast_4s_mean_m": fmean,
        "best_forecast_baseline": fbest,
        "implausible_fraction": {"model": float(c["bad_model"].mean()), "truth": float(c["bad_truth"].mean())},
        "report_only_smoothed_forecast": {
            "forecast_4s_mean_m": smooth_4s,
            "implausible_fraction": float(c["bad_model_smooth"].mean()),
        },  # amendment A1
    }
    res["rules"] = {
        "R1_ratio": med["model"] / med[best],
        "R1_pass": bool(med["model"] <= 0.9 * med[best]),
        "R2_pass": bool(0.85 <= res["coverage_90"] <= 0.95),
        "R3a_pass": bool(fmean["model"] < fmean[fbest]),
        "R3b_pass": bool(res["implausible_fraction"]["model"] <= res["implausible_fraction"]["truth"]),
        "R4": "not tested: the what-if stage was not built",
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/skillcorner")
    ap.add_argument("--dry-run", action="store_true", help="development match only; checks the code, does not count")
    args = ap.parse_args()
    matches, holdout = (DRY_RUN, False) if args.dry_run else (HOLDOUT, True)
    model = AnchoredModel(length=90, dim=128, heads=4, depth=4)
    model.load_state_dict(torch.load(ROOT / "data" / "models" / "05-more-data.pt", map_location="cpu"))
    model.eval()
    per, all_parts = {}, []
    for m, donor in matches.items():
        p = parts(model, *load(m, donor, Path(args.raw), holdout))
        per[f"idsse-{m}"] = summarise([p])
        all_parts.append(p)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    out = {
        "spec": "07-holdout" + (" (dry run, development match, does not count)" if args.dry_run else ""),
        "commit": commit.stdout.strip(),
        "weights": "data/models/05-more-data.pt",
        "std_scale": STD_SCALE,
        "donors": matches,
        "pooled": summarise(all_parts),
        "per_match": per,
    }
    name = "07-dry-run.json" if args.dry_run else "07-holdout.json"
    (ROOT / "results" / name).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["pooled"], indent=2))


if __name__ == "__main__":
    main()
