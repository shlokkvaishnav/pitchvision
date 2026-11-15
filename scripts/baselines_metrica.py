"""Spec 02: imputation baselines on Metrica games with the fitted camera mask.

Usage: PYTHONPATH=src .venv/bin/python scripts/baselines_metrica.py
Needs results/01-camera.json. Writes results/02-baselines.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from pitchvision import baselines as bl
from pitchvision.camera import footprint_sequences, polygon_mask, replay
from pitchvision.splits import assert_not_holdout
from pitchvision.tracking import from_kloppy, load_skillcorner_cached, resample

ROOT = Path(__file__).resolve().parents[1]
FPS = 10.0
BINS = {"lt_1s": (0, 1), "1_3s": (1, 3), "3_10s": (3, 10), "gt_10s": (10, np.inf)}


def replay_mask(tr, donor) -> np.ndarray:
    """Spec 02, amendment A1: the replay_ball mask with a SkillCorner donor match."""
    keep = np.isin(donor.period, (1, 2))
    seqs = footprint_sequences(donor.period[keep], donor.ball[keep, 0], donor.corners[keep])
    return polygon_mask(tr.players, replay(seqs, tr.period, tr.ball[:, 0]))


def evaluate(tr, visible) -> dict:
    hidden = ~visible & np.isfinite(tr.players).all(-1)
    gaps = bl.gap_lengths(hidden) / FPS
    estimates = {
        "hold_last": bl.hold_last(tr.players, visible),
        "constant_velocity": bl.constant_velocity(tr.players, visible, FPS),
        "linear_interpolation": bl.linear_interpolation(tr.players, visible),
        "formation_template": bl.formation_template(tr.players, visible, tr.teams)[0],
        "kalman_smoother": bl.kalman_smoother(tr.players, visible, FPS),
    }
    out = {"hidden_fraction": float(hidden.sum() / np.isfinite(tr.players).all(-1).sum()), "baselines": {}}
    for name, est in estimates.items():
        rec = {"all": bl.errors(est, tr.players, hidden)}
        for b, (lo, hi) in BINS.items():
            sel = hidden & (gaps >= lo) & (gaps < hi)
            rec[b] = bl.errors(est, tr.players, sel) if sel.any() else None
        rec["_abs_errors"] = np.linalg.norm(est[hidden] - tr.players[hidden], axis=-1)
        out["baselines"][name] = rec
    return out


def main() -> None:
    from kloppy import metrica

    sc_dirs = sorted(p for p in (ROOT / "data" / "raw" / "skillcorner").iterdir() if p.is_dir())

    games, pooled = {}, {}
    for g, donor_dir in zip((1, 2, 3), sc_dirs):
        mid = f"metrica-{g}"
        assert_not_holdout(mid)
        ds = metrica.load_open_data(match_id=g, coordinates="secondspectrum")
        tr = resample(from_kloppy(ds, mid), FPS)
        res = evaluate(tr, replay_mask(tr, load_skillcorner_cached(donor_dir)))
        res["donor"] = donor_dir.name
        for name, rec in res["baselines"].items():
            pooled.setdefault(name, []).append(rec.pop("_abs_errors"))
        games[mid] = res
        print(mid, json.dumps({k: v["all"]["median"] for k, v in res["baselines"].items()}), flush=True)

    pooled_median = {k: float(np.median(np.concatenate(v))) for k, v in pooled.items()}
    bar = min(pooled_median, key=pooled_median.get)
    short = [games[m]["baselines"]["linear_interpolation"]["lt_1s"]["median"] for m in games]

    det = []
    for folder in sc_dirs:
        sc = load_skillcorner_cached(folder)
        has = np.isfinite(sc.players).all(-1) & np.isin(sc.period, (1, 2))[:, None]
        det.append((~sc.detected[has]).mean())

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    out = {
        "spec": "02-baselines",
        "commit": sha.stdout.strip(),
        "mask": "replay_ball (spec 01 A2, spec 02 A1); fails Q2 and Q3",
        "fps": FPS,
        "pooled_median_error_m": pooled_median,
        "B1_bar_for_R1": bar,
        "B2_linear_lt_1s_median_m": short,
        "B2_pass": all(s < 1.0 for s in short),
        "skillcorner_undetected_fraction": float(np.mean(det)),
        "metrica_hidden_fraction": float(np.mean([games[m]["hidden_fraction"] for m in games])),
        "games": games,
    }
    (ROOT / "results" / "02-baselines.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != "games"}, indent=2))


if __name__ == "__main__":
    main()
