"""Pitch control from the filled-in positions against the true positions, on every frame of the demo clips.

Exploratory, demo clips only (not a study). Usage: PYTHONPATH=src .venv/bin/python scripts/pitch_control_demo.py
Writes results/pitch_control_demo.json. Hidden players use the model output stored in clips.json (python_mean).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from pitchvision.pitch_control import control_map

ROOT = Path(__file__).resolve().parents[1]


def velocities(series: np.ndarray, f: int, fps: float) -> np.ndarray:
    a, b = max(0, f - 2), min(len(series) - 1, f + 2)
    return (series[b] - series[a]) / ((b - a) / fps)


def main() -> None:
    meta = json.loads((ROOT / "site" / "demo" / "clips.json").read_text())
    team, scale = np.array(meta["agent_type"][:22]), np.array(meta["scale_m"])
    per_clip = []
    for k, c in enumerate(meta["clips"]):
        truth = np.array(c["truth_m"], float)[:, :22]
        seen = np.array(c["visible"], bool)[:, :22]
        est = np.where(seen[..., None], truth, np.array(c["python_mean"], float)[:, :22] * scale)
        same, abs_diff, frames = [], [], len(truth)
        for f in range(frames):
            ball = np.array(c["truth_m"][f][22], float)
            mt = control_map(truth[f], velocities(truth, f, meta["fps"]), team, ball)
            me = control_map(est[f], velocities(est, f, meta["fps"]), team, ball)
            same.append(float(np.mean((mt >= 0.5) == (me >= 0.5))))
            abs_diff.append(float(np.mean(np.abs(mt - me))))
        per_clip.append(
            {
                "clip": k,
                "frames": frames,
                "hidden_share": float(1 - seen.mean()),
                "same_team_share": float(np.mean(same)),
                "worst_frame_same_team_share": float(np.min(same)),
                "mean_abs_diff": float(np.mean(abs_diff)),
            }
        )
        print(per_clip[-1], flush=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    out = {
        "commit": sha,
        "grid_cols": 32,
        "note": "exploratory, demo clips only",
        "clips": per_clip,
        "same_team_share_mean": float(np.mean([c["same_team_share"] for c in per_clip])),
    }
    (ROOT / "results" / "pitch_control_demo.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
