"""Spec 08 part A: spread of the spec 05 test metrics over training seeds 0, 1 and 2.

Usage: python scripts/seeds_summary.py
Reads results/05-more-data.json (seed 0) and results/08-seeds/seed{1,2}.json (Kaggle runs).
Writes results/08-seeds.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["imputation_median_m", "kalman_same_cells_median_m", "coverage_90", "forecast_4s_mean_m",
           "constant_velocity_4s_mean_m"]  # fmt: skip


def main() -> None:
    files = {0: ROOT / "results" / "05-more-data.json"}
    files.update({s: ROOT / "results" / "08-seeds" / f"seed{s}.json" for s in (1, 2)})
    runs = {s: json.loads(p.read_text()) for s, p in files.items()}
    per_seed = {
        s: {**{m: r["test"][m] for m in METRICS}, "ratio_vs_kalman": r["rules"]["M1_ratio"],
            "best_epoch": r["best_epoch"], "scale_factor": r["scale_factor"],
            "all_rules_pass": all(v for k, v in r["rules"].items() if k.endswith("_pass"))}
        for s, r in runs.items()
    }  # fmt: skip
    keys = [*METRICS, "ratio_vs_kalman", "scale_factor"]
    summary = {
        k: {"mean": sum(p[k] for p in per_seed.values()) / len(per_seed),
            "min": min(p[k] for p in per_seed.values()), "max": max(p[k] for p in per_seed.values())}
        for k in keys
    }  # fmt: skip
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    out = {"spec": "08-seeds-and-intervals (part A)", "test_matches": ["J03WQQ", "J03WR9"],
           "seeds": per_seed, "summary": summary, "commit": sha.stdout.strip()}  # fmt: skip
    (ROOT / "results" / "08-seeds.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
