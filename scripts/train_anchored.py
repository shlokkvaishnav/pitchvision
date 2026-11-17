"""Spec 04: train masked model version B (Kalman anchor) on Metrica games 1-2, validate on game 3.

Usage: PYTHONPATH=src python scripts/train_anchored.py --out results [--raw data/raw/skillcorner] [--epochs 30]
Runs on CPU or GPU. On Kaggle it runs from the kernel in kaggle/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from pitchvision.dataset import FIELDS, donors, load_match
from pitchvision.metrics import match_cell_errors, pooled
from pitchvision.model import AnchoredModel
from pitchvision.train import fit


def evaluate(model, tr, vis, w, device) -> dict:
    """Spec 04 metrics on one validation match."""
    return {"n_windows": len(w.pos), **pooled([match_cell_errors(model, tr, vis, w, device)])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--raw", default="data/raw/skillcorner")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0, help="smoke test only: limit train and validation windows")
    ap.add_argument("--commit", default=os.environ.get("PITCHVISION_COMMIT", "unknown"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dn = donors()
    games = {g: load_match(f"metrica-{g}", dn[f"metrica-{g}"], Path(args.raw)) for g in (1, 2, 3)}
    train_w, w2 = games[1][2], games[2][2]
    for f in FIELDS:
        setattr(train_w, f, np.concatenate([getattr(train_w, f), getattr(w2, f)]))
    if args.max_windows:
        for w in (train_w, games[3][2]):
            for f in FIELDS:
                setattr(w, f, getattr(w, f)[: args.max_windows])

    model = AnchoredModel(length=90, dim=128, heads=4, depth=4).to(device)
    t0 = time.time()
    log = [e["train_loss"] for e in fit(model, train_w, device, args.epochs, args.batch, args.lr, args.seed)[0]]
    tr3, vis3, w3 = games[3]
    m = evaluate(model, tr3, vis3, w3, device)
    rules = {
        "M1_ratio": m["imputation_median_m"] / m["kalman_same_cells_median_m"],
        "M1_pass": bool(m["imputation_median_m"] <= 0.9 * m["kalman_same_cells_median_m"]),
        "M2_pass": bool(0.85 <= m["coverage_90"] <= 0.95),
        "M3_pass": bool(m["forecast_4s_mean_m"] < m["constant_velocity_4s_mean_m"]),
    }
    out = {
        "spec": "04-anchored-model",
        "commit": args.commit,
        "device": device,
        "train_seconds": round(time.time() - t0, 1),
        "settings": vars(args),
        "train_windows": len(train_w.pos),
        "loss_per_epoch": log,
        "validation_game3": m,
        "rules": rules,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "04-anchored-model.json").write_text(json.dumps(out, indent=2))
    torch.save(model.state_dict(), Path(args.out) / "04-anchored-model.pt")
    print(json.dumps({"validation_game3": m, "rules": rules}, indent=2))


if __name__ == "__main__":
    main()
