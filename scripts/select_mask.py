"""Spec 01, amendment A2: select the camera mask by distribution rules Q1-Q3.

Usage: PYTHONPATH=src .venv/bin/python scripts/select_mask.py
Writes results/01-mask-selection.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from pitchvision.camera import Footprint, camera_centre, footprint_sequences, polygon_mask, replay, visible_mask
from pitchvision.tracking import load_skillcorner_cached

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "skillcorner"
BANDS = {"near_lt_-20": (-np.inf, -20.0), "middle": (-20.0, 20.0), "far_gt_20": (20.0, np.inf)}


def run_lengths(hidden: np.ndarray) -> np.ndarray:
    """Lengths of all runs of True along axis 0, for each column."""
    out = []
    for j in range(hidden.shape[1]):
        edges = np.flatnonzero(np.diff(np.concatenate([[0], hidden[:, j].astype(int), [0]])))
        out.extend(edges[1::2] - edges[::2])
    return np.asarray(out)


def centroid_x(tr) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(tr.players[..., 0], axis=1)


def masks(tr, donor, trapezoid: Footprint) -> dict[str, np.ndarray]:
    centre = np.empty(len(tr.period))
    for p in np.unique(tr.period):
        rows = tr.period == p
        centre[rows] = camera_centre(tr.ball[rows, 0], trapezoid, tr.pitch[0])
    seq_ball = footprint_sequences(donor.period, donor.ball[:, 0], donor.corners)
    seq_cent = footprint_sequences(donor.period, centroid_x(donor), donor.corners)
    return {
        "trapezoid": visible_mask(tr.players, tr.ball[:, 0], trapezoid, tr.pitch, centre=centre),
        "replay_ball": polygon_mask(tr.players, replay(seq_ball, tr.period, tr.ball[:, 0])),
        "replay_centroid": polygon_mask(tr.players, replay(seq_cent, tr.period, centroid_x(tr))),
    }


def main() -> None:
    cam = json.loads((ROOT / "results" / "01-camera.json").read_text())["final_params_all_matches"]
    trapezoid = Footprint(cam["near_width"], cam["far_width"], cam["pan_smoothing"])
    ids = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    matches = {}
    for mid in ids:
        tr = load_skillcorner_cached(RAW / mid)
        matches[mid] = tr.select(np.flatnonzero(np.isin(tr.period, (1, 2))))

    names = ("trapezoid", "replay_ball", "replay_centroid")
    acc = {
        n: {
            "hidden": 0,
            "cells": 0,
            "agree": 0,
            "agree_cells": 0,
            "runs": [],
            "band_hidden": dict.fromkeys(BANDS, 0),
            "band_cells": dict.fromkeys(BANDS, 0),
        }
        for n in (*names, "real")
    }
    for i, mid in enumerate(ids):
        tr, donor = matches[mid], matches[ids[(i + 1) % len(ids)]]
        valid = np.isfinite(tr.players).all(-1)
        real_hidden = ~tr.detected & valid
        sim = {n: ~m & valid for n, m in masks(tr, donor, trapezoid).items()}
        sim["real"] = real_hidden
        y = tr.players[..., 1]
        sec = np.zeros(len(tr.period), dtype=bool)
        sec[::10] = True
        for n, hidden in sim.items():
            a = acc[n]
            a["hidden"] += int(hidden.sum())
            a["cells"] += int(valid.sum())
            a["agree"] += int(((hidden == real_hidden) & valid)[sec].sum())
            a["agree_cells"] += int(valid[sec].sum())
            a["runs"].append(run_lengths(hidden))
            for b, (lo, hi) in BANDS.items():
                in_band = valid & (y >= lo) & (y < hi)
                a["band_hidden"][b] += int((hidden & in_band).sum())
                a["band_cells"][b] += int(in_band.sum())
        print("done", mid, flush=True)

    real = acc["real"]
    real_frac = real["hidden"] / real["cells"]
    real_runs = np.concatenate(real["runs"])
    real_band = {b: real["band_hidden"][b] / real["band_cells"][b] for b in BANDS}
    results = {}
    for n in names:
        a = acc[n]
        frac = a["hidden"] / a["cells"]
        ks = float(ks_2samp(np.concatenate(a["runs"]), real_runs).statistic)
        band = {b: a["band_hidden"][b] / a["band_cells"][b] for b in BANDS}
        results[n] = {
            "hidden_fraction": frac,
            "ks_run_lengths": ks,
            "band_hidden_fraction": band,
            "C1_agreement": a["agree"] / a["agree_cells"],
            "Q1": abs(frac - real_frac) <= 0.03,
            "Q2": ks <= 0.10,
            "Q3": all(abs(band[b] - real_band[b]) <= 0.05 for b in BANDS),
        }
        results[n]["n_pass"] = int(results[n]["Q1"]) + int(results[n]["Q2"]) + int(results[n]["Q3"])
    selected = max(names, key=lambda n: (results[n]["n_pass"] == 3, results[n]["n_pass"], results[n]["C1_agreement"]))

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    out = {
        "spec": "01-data-and-camera (A2 mask selection)",
        "commit": sha.stdout.strip(),
        "real": {
            "hidden_fraction": real_frac,
            "band_hidden_fraction": real_band,
            "median_run_s": float(np.median(real_runs)) / 10,
        },
        "candidates": results,
        "selected": selected,
        "selected_passes_all": results[selected]["n_pass"] == 3,
    }
    (ROOT / "results" / "01-mask-selection.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
