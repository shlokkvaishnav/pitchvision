"""Spec 08 part B: 95% block-bootstrap intervals for the seed 0 model (spec 05 test matches and spec 07 locked matches).

Usage: PYTHONPATH=src python scripts/intervals.py
Needs data/models/05-more-data.pt. Writes results/08-intervals.json. CPU only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from pitchvision import baselines as bl
from pitchvision.anchor import SCALE
from pitchvision.dataset import HORIZON, load_match
from pitchvision.metrics import at_cells, cell_scores
from pitchvision.model import AnchoredModel, predict_windows
from pitchvision.prep import FPS

ROOT = Path(__file__).resolve().parents[1]
STD_SCALE = 1.1747  # spec 05
SETS = {
    "spec05_test": ({"J03WQQ": "1959846", "J03WR9": "1986691"}, False),
    "spec07_locked": ({"J03WN1": "1996435", "J03WOH": "1996436"}, True),
}
BLOCK_FRAMES = 5 * 60 * FPS  # 5-minute blocks
N_BOOT, BIN_CM, MAX_CM = 2000, 1, 20_000


@torch.no_grad()
def blocks_of_match(model, name: str, donor: str, raw: Path, allow: bool) -> dict[str, dict]:
    """Per 5-minute block: histograms of imputation errors (model, spline), coverage and forecast sums."""
    tr, vis, w = load_match(f"idsse-{name}", donor, raw, allow_holdout=allow)
    hidden = ~w.visible[..., :22]
    block_w = (w.start // BLOCK_FRAMES).astype(int)
    block_cell = np.broadcast_to(block_w[:, None, None], hidden.shape)[hidden]

    mu, ls = predict_windows(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, "cpu")
    err, inside = cell_scores(mu, ls + np.log(STD_SCALE), w, hidden)
    spline = at_cells(bl.by_period(bl.cubic_spline, tr.players, vis, tr.period), tr, w, hidden)
    ok = np.isfinite(err) & np.isfinite(spline)

    obs_f = w.visible.copy()
    obs_f[:, -HORIZON:, :] = False
    mu_f, _ = predict_windows(model, w.pos, obs_f, w.agent_type, w.fc_mean, w.fc_ls, "cpu")
    truth = w.pos[:, -1, :22] * SCALE
    fc = np.linalg.norm(mu_f[:, -1, :22] * SCALE - truth, axis=-1)
    cv = np.stack(
        [bl.constant_velocity(w.pos[b, :, :22] * SCALE, obs_f[b, :, :22], FPS)[-1] for b in range(len(w.pos))]
    )
    cv_err = np.linalg.norm(cv - truth, axis=-1)
    fok = np.isfinite(cv_err) & np.isfinite(fc)
    block_fc = np.broadcast_to(block_w[:, None], fc.shape)

    def hist(x):
        return np.bincount(np.clip((x * 100 / BIN_CM).astype(int), 0, MAX_CM), minlength=MAX_CM + 1)

    out = {}
    for b in np.unique(block_w):
        c, f = ok & (block_cell == b), fok & (block_fc == b)
        out[f"{name}-{b}"] = {
            "h_model": hist(err[c]), "h_spline": hist(spline[c]),
            "inside": float(inside[np.isfinite(err) & (block_cell == b)].sum()),
            "n_cov": int((np.isfinite(err) & (block_cell == b)).sum()),
            "fc": float(fc[f].sum()), "cv": float(cv_err[f].sum()), "n_fc": int(f.sum()),
        }  # fmt: skip
    return out


def summarise(blocks: dict[str, dict]) -> dict:
    keys = list(blocks)
    hm = np.stack([blocks[k]["h_model"] for k in keys]).astype(float)
    hs = np.stack([blocks[k]["h_spline"] for k in keys]).astype(float)
    sc = np.array([[blocks[k][f] for f in ("inside", "n_cov", "fc", "cv", "n_fc")] for k in keys])
    rng = np.random.default_rng(0)
    weights = np.vstack([np.ones(len(keys)), rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), N_BOOT)])

    def median(h):  # rows of histograms -> medians in metres (bin centre)
        cum = np.cumsum(h, axis=1)
        return (np.argmax(cum >= cum[:, -1:] / 2, axis=1) + 0.5) * BIN_CM / 100

    med_m, med_s = median(weights @ hm), median(weights @ hs)
    s = weights @ sc
    q = {
        "model_median_m": med_m,
        "spline_median_m": med_s,
        "median_ratio": med_m / med_s,
        "coverage_90": s[:, 0] / s[:, 1],
        "forecast_4s_mean_m": s[:, 2] / s[:, 4],
        "cv_last_4s_mean_m": s[:, 3] / s[:, 4],
        "forecast_minus_cv_m": (s[:, 2] - s[:, 3]) / s[:, 4],
    }
    return {
        "n_blocks": len(keys),
        **{k: {"estimate": float(v[0]), "ci95": [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]}
           for k, v in q.items()},
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/skillcorner")
    args = ap.parse_args()
    model = AnchoredModel(length=90, dim=128, heads=4, depth=4)
    model.load_state_dict(torch.load(ROOT / "data" / "models" / "05-more-data.pt", map_location="cpu"))
    out: dict = {"spec": "08-seeds-and-intervals (part B)", "n_boot": N_BOOT, "block_minutes": 5}
    for name, (matches, allow) in SETS.items():
        blocks: dict[str, dict] = {}
        for m, donor in matches.items():
            blocks.update(blocks_of_match(model, m, donor, Path(args.raw), allow))
        out[name] = summarise(blocks)
        print(name, json.dumps(out[name], indent=2), flush=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    out["commit"] = sha.stdout.strip()
    (ROOT / "results" / "08-intervals.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
