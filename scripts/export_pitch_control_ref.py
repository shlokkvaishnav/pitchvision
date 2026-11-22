"""Write tests/js/pitch-control-ref.json: Python pitch control on demo frames, for the JavaScript parity test.

Usage: PYTHONPATH=src .venv/bin/python scripts/export_pitch_control_ref.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pitchvision.pitch_control import control_map

ROOT = Path(__file__).resolve().parents[1]
CASES = [(0, 0), (0, 45), (1, 60)]  # (clip, frame)


def velocities(series: np.ndarray, f: int, fps: float) -> np.ndarray:
    a, b = max(0, f - 2), min(len(series) - 1, f + 2)
    return (series[b] - series[a]) / ((b - a) / fps)


def main() -> None:
    meta = json.loads((ROOT / "site" / "demo" / "clips.json").read_text())
    team = np.array(meta["agent_type"][:22])
    out = []
    for c, f in CASES:
        truth = np.array(meta["clips"][c]["truth_m"], float)
        pos, vel, ball = truth[f, :22], velocities(truth[:, :22], f, meta["fps"]), truth[f, 22]
        home = control_map(pos, vel, team, ball, nx=32)
        out.append({"clip": c, "frame": f, "home": np.round(home.ravel(), 8).tolist()})
    path = ROOT / "tests" / "js" / "pitch-control-ref.json"
    path.write_text(json.dumps({"nx": 32, "cases": out}))
    print("wrote", path, sum(len(o["home"]) for o in out), "cells")


if __name__ == "__main__":
    main()
