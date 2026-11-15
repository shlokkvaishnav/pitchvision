"""Spec 01, amendment A1: test the footprint replay across SkillCorner matches.

Usage: PYTHONPATH=src .venv/bin/python scripts/eval_replay.py
Writes results/01-replay.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from pitchvision.camera import footprint_sequences, polygon_mask, replay
from pitchvision.tracking import load_skillcorner_cached

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "skillcorner"


def score(pred: np.ndarray, det: np.ndarray, valid: np.ndarray) -> dict:
    det = det & valid
    pred = pred & valid
    return {
        "agreement": float((pred == det)[valid].mean()),
        "count_mae": float(np.abs(pred.sum(1) - det.sum(1)).mean()),
        "n": int(valid.sum()),
    }


def main() -> None:
    ids = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    matches = {}
    for mid in ids:
        tr = load_skillcorner_cached(RAW / mid)
        keep = np.flatnonzero(np.isin(tr.period, (1, 2)))
        matches[mid] = tr.select(keep)
        print("loaded", mid, flush=True)

    per_match = []
    for i, mid in enumerate(ids):
        tr, donor = matches[mid], matches[ids[(i + 1) % len(ids)]]
        seqs = footprint_sequences(donor.period, donor.ball[:, 0], donor.corners)
        rows = np.arange(0, len(tr.period), 10)  # one frame per second
        fake = replay(seqs, tr.period, tr.ball[:, 0])[rows]
        pos = tr.players[rows]
        valid = np.isfinite(pos).all(-1)
        rec = {
            "match_id": mid,
            "donor": donor.match_id,
            "replay": score(polygon_mask(pos, fake), tr.detected[rows], valid),
            "reference": score(polygon_mask(pos, tr.corners[rows]), tr.detected[rows], valid),
            "undetected_fraction": float((~tr.detected[rows])[valid].mean()),
        }
        per_match.append(rec)
        print(json.dumps(rec), flush=True)

    w = np.array([r["replay"]["n"] for r in per_match], dtype=float)
    agree = float(np.average([r["replay"]["agreement"] for r in per_match], weights=w))
    ref = float(np.average([r["reference"]["agreement"] for r in per_match], weights=w))
    count = float(np.mean([r["replay"]["count_mae"] for r in per_match]))
    rules = {
        "agreement": agree,
        "reference_agreement": ref,
        "count_mae": count,
        "C1_agreement_ge_0.85": agree >= 0.85,
        "C2_within_5pp_of_reference": agree >= ref - 0.05,
        "C3_count_mae_le_1.0": count <= 1.0,
        "undetected_fraction": float(np.average([r["undetected_fraction"] for r in per_match], weights=w)),
    }
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    out = {"spec": "01-data-and-camera (A1 replay)", "commit": sha.stdout.strip(), "rules": rules, "matches": per_match}
    (ROOT / "results" / "01-replay.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(rules, indent=2))


if __name__ == "__main__":
    main()
