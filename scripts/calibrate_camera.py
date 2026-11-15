"""Spec 01: download SkillCorner open data, fit the camera mask, write results/01-camera.json.

Usage: PYTHONPATH=src .venv/bin/python scripts/calibrate_camera.py
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

from pitchvision.calibrate import cross_validate, fit, prepare, rules
from pitchvision.tracking import SKILLCORNER_MEDIA, SKILLCORNER_RAW, load_skillcorner

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "skillcorner"
MATCHES_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches.json"


def download(match_id: str) -> tuple[Path, Path]:
    folder = RAW / match_id
    folder.mkdir(parents=True, exist_ok=True)
    tracking, match = folder / "tracking.jsonl", folder / "match.json"
    if not tracking.exists() or tracking.stat().st_size < 10_000:
        urllib.request.urlretrieve(f"{SKILLCORNER_MEDIA}/{match_id}/{match_id}_tracking_extrapolated.jsonl", tracking)
    if not match.exists():
        urllib.request.urlretrieve(f"{SKILLCORNER_RAW}/{match_id}/{match_id}_match.json", match)
    return tracking, match


def main() -> None:
    with urllib.request.urlopen(MATCHES_URL) as r:
        ids = sorted(str(m["id"]) for m in json.load(r))
    prepared = []
    for mid in ids:
        tr = load_skillcorner(*download(mid))
        prepared.append(prepare(tr, score_every=10))  # 10 Hz data -> one frame per second
        print(f"loaded {mid}: {tr.players.shape[0]} frames", flush=True)

    folds = cross_validate(prepared, k=5)
    near, far, a = fit(prepared)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    out = {
        "spec": "01-data-and-camera",
        "commit": sha,
        "matches": ids,
        "rules": rules(folds),
        "final_params_all_matches": {"near_width": near, "far_width": far, "pan_smoothing": a},
        "folds": folds,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "01-camera.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["rules"], indent=2))
    print(json.dumps(out["final_params_all_matches"]))


if __name__ == "__main__":
    main()
