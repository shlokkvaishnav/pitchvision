"""Spec 05: masked model B with more data (IDSSE), early stopping and calibration.

Usage: PYTHONPATH=src python scripts/train_more.py --out results [--raw data/raw/skillcorner] [--epochs 30]
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

from pitchvision.dataset import concat, donors, load_match
from pitchvision.metrics import match_cell_errors, pooled
from pitchvision.model import AnchoredModel
from pitchvision.train import fit

TRAIN = ["metrica-1", "metrica-2", "idsse-J03WMX", "idsse-J03WOY", "idsse-J03WPY"]
TUNE = ["metrica-3"]
TEST = ["idsse-J03WQQ", "idsse-J03WR9"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--raw", default="data/raw/skillcorner")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0, help="smoke test only: limit windows per match")
    ap.add_argument("--smoke", action="store_true", help="smoke test only: Metrica games, no IDSSE")
    ap.add_argument("--commit", default=os.environ.get("PITCHVISION_COMMIT", "unknown"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw, dn = Path(args.raw), donors()
    train_ids, tune_ids, test_ids = (["metrica-1"], ["metrica-3"], ["metrica-2"]) if args.smoke else (TRAIN, TUNE, TEST)
    data = {m: load_match(m, dn[m], raw, args.max_windows) for m in train_ids + tune_ids + test_ids}
    train_w = concat([data[m][2] for m in train_ids])
    tune = [data[m] for m in tune_ids]

    model = AnchoredModel(length=90, dim=128, heads=4, depth=4).to(device)
    t0 = time.time()
    log, best_epoch = fit(model, train_w, device, args.epochs, args.batch, args.lr, args.seed, tune=tune)

    # calibration: one scale factor on the tuning match (closed form for a Gaussian)
    z2 = np.concatenate([match_cell_errors(model, *data[m], device)["z2"].ravel() for m in tune_ids])
    log_scale = float(0.5 * np.log(np.mean(z2)))
    tune_m = pooled([match_cell_errors(model, *data[m], device, log_scale) for m in tune_ids])
    test_m = pooled([match_cell_errors(model, *data[m], device, log_scale) for m in test_ids])
    rules = {
        "M1_ratio": test_m["imputation_median_m"] / test_m["kalman_same_cells_median_m"],
        "M1_pass": bool(test_m["imputation_median_m"] <= 0.9 * test_m["kalman_same_cells_median_m"]),
        "M2_pass": bool(0.85 <= test_m["coverage_90"] <= 0.95),
        "M3_pass": bool(test_m["forecast_4s_mean_m"] < test_m["constant_velocity_4s_mean_m"]),
    }
    out = {
        "spec": "05-more-data",
        "commit": args.commit,
        "device": device,
        "train_seconds": round(time.time() - t0, 1),
        "settings": vars(args),
        "windows": {m: len(data[m][2].pos) for m in data},
        "best_epoch": best_epoch,
        "scale_factor": float(np.exp(log_scale)),
        "log": log,
        "tuning_metrica3": tune_m,
        "test": test_m,
        "rules": rules,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "05-more-data.json").write_text(json.dumps(out, indent=2))
    torch.save(model.state_dict(), Path(args.out) / "05-more-data.pt")
    print(json.dumps({"test": test_m, "rules": rules, "best_epoch": best_epoch}, indent=2))


if __name__ == "__main__":
    main()
