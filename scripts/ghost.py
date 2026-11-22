"""Spec 10: ghost team, leave-one-match-out over the 8 development matches (exploratory).

Usage: PYTHONPATH=src .venv/bin/python scripts/ghost.py [--epochs 8]
Writes results/10-ghost.json and data/models/ghost.pt (a model trained on all 8 matches, for the site).
The spec 07 holdout matches are never loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from pitchvision.ghost import (
    Samples,
    fit_linear_shift,
    inside_90,
    predict_ghost,
    predict_linear_shift,
    samples,
    train_ghost,
)
from pitchvision.prep import IDSSE_DEV, idsse_match, metrica_game

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "ghost"
MATCHES = [f"metrica-{g}" for g in (1, 2, 3)] + [f"idsse-{m}" for m in IDSSE_DEV]


def load(name: str) -> Samples:
    path = CACHE / f"{name}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return Samples(z["other"], z["ball"], z["role"], z["target"], z["block"])
    tr = metrica_game(int(name.split("-")[1])) if name.startswith("metrica") else idsse_match(name.split("-")[1])
    s = samples(tr)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, other=s.other, ball=s.ball, role=s.role, target=s.target, block=s.block)
    return s


def bootstrap_ratio(blocks: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> list[float]:
    """95% interval of mean(a) / mean(b), resampling blocks."""
    keys, inv = np.unique(blocks, return_inverse=True)
    sa, sb, cnt = (np.bincount(inv, weights=w, minlength=len(keys)) for w in (a, b, np.ones_like(a)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n, len(keys)))
    r = (sa[idx].sum(1) / cnt[idx].sum(1)) / (sb[idx].sum(1) / cnt[idx].sum(1))
    return [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = {}
    for m in MATCHES:
        data[m] = load(m)
        print(m, len(data[m]), "samples", flush=True)

    err = {k: [] for k in ("role", "linear_shift", "ghost")}
    cover, blocks, per_match = [], [], []
    for m in MATCHES:
        t0 = time.time()
        train = Samples.concat([data[x] for x in MATCHES if x != m])
        test = data[m]
        w = fit_linear_shift(train)
        model = train_ghost(train, device, epochs=args.epochs)
        mean, std = predict_ghost(model, test, device)
        preds = {"role": test.role, "linear_shift": predict_linear_shift(w, test), "ghost": mean}
        row = {"match": m, "samples": len(test)}
        for k, p in preds.items():
            e = np.linalg.norm(p - test.target, axis=-1).ravel()
            err[k].append(e)
            row[f"{k}_mean_m"] = float(e.mean())
        c = inside_90(mean, std, test.target).ravel()
        cover.append(c)
        row["ghost_cover_90"] = float(c.mean())
        blocks.append(np.repeat(test.block, 11))
        per_match.append(row)
        print(row, f"{time.time() - t0:.0f} s", flush=True)

    e = {k: np.concatenate(v) for k, v in err.items()}
    bl = np.concatenate(blocks)
    cov = float(np.concatenate(cover).mean())
    ratio = float(e["ghost"].mean() / e["linear_shift"].mean())
    ci = bootstrap_ratio(bl, e["ghost"], e["linear_shift"])
    rules = {
        "G1_ratio": ratio,
        "G1_ci": ci,
        "G1_pass": bool(ratio <= 0.90 and ci[1] < 1),
        "G2_cover_90": cov,
        "G2_pass": bool(0.85 <= cov <= 0.95),
    }
    commit = (
        os.environ.get("PITCHVISION_COMMIT")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    out = {
        "spec": "10-ghost-team",
        "commit": commit,
        "exploratory": "leave-one-match-out over the 8 development matches; holdout not loaded",
        "epochs": args.epochs,
        "pooled": {k: {"mean_m": float(v.mean()), "median_m": float(np.median(v))} for k, v in e.items()},
        "rules": rules,
        "matches": per_match,
    }
    (ROOT / "results" / "10-ghost.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("pooled", "rules")}, indent=2))

    final = train_ghost(Samples.concat(list(data.values())), device, epochs=args.epochs)
    (ROOT / "data" / "models").mkdir(parents=True, exist_ok=True)
    torch.save(final.state_dict(), ROOT / "data" / "models" / "ghost.pt")


if __name__ == "__main__":
    main()
