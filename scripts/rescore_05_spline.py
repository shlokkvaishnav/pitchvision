"""Spec 05 note: compare the spec 05 model with the cubic spline (spec 06 bar) on the same test cells.

Usage: PYTHONPATH=src python scripts/rescore_05_spline.py [--raw data/raw/skillcorner]
Needs data/models/05-more-data.pt. Writes results/05-spline-same-cells.json. CPU only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_anchored as ta
import train_more as tm

from pitchvision import baselines as bl
from pitchvision.model import AnchoredModel
from pitchvision.prep import FPS

ROOT = Path(__file__).resolve().parents[1]


def _at(est: np.ndarray, tr, w, hidden: np.ndarray) -> np.ndarray:
    frames = np.broadcast_to(w.start[:, None, None] + np.arange(w.pos.shape[1])[None, :, None], hidden.shape)
    cols = np.broadcast_to(w.cols[:, None, :], hidden.shape)
    f, c = frames[hidden], cols[hidden]
    return np.linalg.norm(est[f, c] - tr.players[f, c], axis=-1)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/skillcorner")
    args = ap.parse_args()
    model = AnchoredModel(length=90, dim=128, heads=4, depth=4)
    model.load_state_dict(torch.load(ROOT / "data" / "models" / "05-more-data.pt", map_location="cpu"))
    dn = tm.donors()
    parts: dict[str, list[np.ndarray]] = {"model": [], "kalman": [], "spline": []}
    for name in tm.TEST:
        tr, vis, w = tm.load_match(name, dn[name], Path(args.raw))
        hidden = ~w.visible[..., :22]
        mu, ls = ta.predict(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, "cpu")
        parts["model"].append(ta._cells(mu, ls, w, hidden)[0])
        parts["kalman"].append(_at(bl.kalman_smoother(tr.players, vis, FPS), tr, w, hidden))
        parts["spline"].append(_at(bl.by_period(bl.cubic_spline, tr.players, vis, tr.period), tr, w, hidden))
    err = {k: np.concatenate(v) for k, v in parts.items()}
    keep = np.isfinite(err["kalman"]) & np.isfinite(err["spline"])
    out = {
        "spec": "05-more-data note (spline bar of spec 06)",
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "test_matches": tm.TEST,
        "cell_fraction": float(keep.mean()),
        "median_m": {k: float(np.median(v[keep])) for k, v in err.items()},
        "mean_m": {k: float(np.mean(v[keep])) for k, v in err.items()},
    }
    out["model_beats_spline_median"] = out["median_m"]["model"] < out["median_m"]["spline"]
    out["model_beats_spline_mean"] = out["mean_m"]["model"] < out["mean_m"]["spline"]
    (ROOT / "results" / "05-spline-same-cells.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
