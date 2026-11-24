"""Spec 15: pool the 10 leave-one-match-out folds into results/15-lomo.json (per match, pooled, intervals, claims).

Usage: PYTHONPATH=src python scripts/lomo_summary.py [--dir results/15-lomo] [--allow-missing]
Reads fold-<match>.json and fold-<match>.npz (downloaded from the Kaggle kernels). CPU, about a minute.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from pitchvision.lomo import GAP_BUCKETS_S, GAP_LABELS, MATCHES

ROOT = Path(__file__).resolve().parents[1]
POINT = ("spline", "kalman", "centroid_vote", "transformer", "diffusion")
CLASSICAL = ("spline", "kalman", "centroid_vote")
DIST = ("kalman", "transformer", "diffusion")
PC = ("visible", "spline", "kalman", "centroid_vote", "transformer", "diffusion")
N_BOOT, BIN_CM, MAX_CM = 2000, 1, 20_000


def hist(x: np.ndarray) -> np.ndarray:
    return np.bincount(np.clip((x * 100 / BIN_CM).astype(int), 0, MAX_CM), minlength=MAX_CM + 1)


def median_of(h: np.ndarray) -> np.ndarray:
    """Rows of histograms -> medians in metres (bin centre)."""
    cum = np.cumsum(h, axis=-1)
    return (np.argmax(cum >= cum[..., -1:] / 2, axis=-1) + 0.5) * BIN_CM / 100


def ci(v: np.ndarray) -> dict:
    return {"estimate": float(v[0]), "ci95": [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]}


class Boot:
    """Stratified block bootstrap: row 0 is the data, rows 1.. resample the blocks within each match."""

    def __init__(self, match_of_block: np.ndarray, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.match_of_block = match_of_block
        self.w = np.zeros((N_BOOT + 1, len(match_of_block)))
        self.w[0] = 1
        for m in np.unique(match_of_block):
            idx = np.flatnonzero(match_of_block == m)
            self.w[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), N_BOOT)

    def sums(self, per_block: np.ndarray) -> np.ndarray:
        """(n_blocks, ...) -> (N_BOOT + 1, ...) weighted sums over all blocks."""
        return np.tensordot(self.w, per_block, axes=(1, 0))

    def sums_by_match(self, per_block: np.ndarray) -> list[np.ndarray]:
        out = []
        for m in np.unique(self.match_of_block):
            idx = np.flatnonzero(self.match_of_block == m)
            out.append(np.tensordot(self.w[:, idx], per_block[idx], axes=(1, 0)))
        return out


def block_index(names: list[str], blocks: list[np.ndarray]):
    """Global block id per cell and the match of each global block."""
    ids, owner, offset = [], [], 0
    for i, b in enumerate(blocks):
        u, inv = np.unique(b, return_inverse=True)
        ids.append(inv + offset)
        owner.extend([i] * len(u))
        offset += len(u)
    return ids, np.asarray(owner)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "15-lomo"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--allow-missing", action="store_true", help="summarise the folds that exist (not a result)")
    args = ap.parse_args()
    folder = Path(args.dir)
    names = [m for m in MATCHES if (folder / f"fold-{m}.npz").exists()]
    if len(names) < len(MATCHES) and not args.allow_missing:
        raise SystemExit(f"missing folds: {sorted(set(MATCHES) - set(names))}")
    js = {m: json.loads((folder / f"fold-{m}.json").read_text()) for m in names}
    zs = {m: dict(np.load(folder / f"fold-{m}.npz")) for m in names}
    commits = sorted({js[m]["commit"] for m in names})

    cell_blocks, owner = block_index(names, [zs[m]["block"] for m in names])
    boot = Boot(owner)
    nb = len(owner)

    # point error: histograms per block and method
    def block_hists(key):
        out = np.zeros((nb, MAX_CM + 1))
        for i, m in enumerate(names):
            x, bi = zs[m][key], cell_blocks[i]
            for b in np.unique(bi):
                out[b] = hist(x[bi == b])
        return out

    hists = {p: block_hists(f"err_{p}") for p in [*POINT, "diffusion_median"]}
    pos_med = {p: median_of(boot.sums(h)) for p, h in hists.items()}
    match_med = {p: np.mean([median_of(s) for s in boot.sums_by_match(h)], axis=0) for p, h in hists.items()}
    best = min(CLASSICAL, key=lambda p: pos_med[p][0])

    def block_sum(key, sel=None):
        out = np.zeros(nb)
        for i, m in enumerate(names):
            x = zs[m][key].astype(float)
            keep = np.ones(len(x), bool) if sel is None else sel(zs[m])
            out += np.bincount(cell_blocks[i][keep], x[keep], nb)
        return out

    count = block_sum_count(zs, names, cell_blocks, nb, lambda z: np.ones(len(z["gap_s"]), bool))

    def bucket(g):
        return lambda z: np.digitize(z["gap_s"], GAP_BUCKETS_S[1:-1]) == g

    pooled: dict = {
        "median_m_position_weighted": {p: ci(v) for p, v in pos_med.items()},
        "median_m_match_weighted": {p: ci(v) for p, v in match_med.items()},
        "best_classical": best,
        "ratio_diffusion_to_best_classical": {
            "position_weighted": ci(pos_med["diffusion"] / pos_med[best]),
            "match_weighted": ci(match_med["diffusion"] / match_med[best]),
        },
        "ratio_diffusion_to_transformer": {
            "position_weighted": ci(pos_med["diffusion"] / pos_med["transformer"]),
            "match_weighted": ci(match_med["diffusion"] / match_med["transformer"]),
        },
        "mean_m": {p: ci(boot.sums(block_sum(f"err_{p}")) / boot.sums(count)) for p in POINT},
        "energy_score_m": {p: ci(boot.sums(block_sum(f"es_{p}")) / boot.sums(count)) for p in DIST},
        "coverage_90": {p: ci(boot.sums(block_sum(f"inside_{p}")) / boot.sums(count)) for p in DIST},
        "coverage_90_by_gap": {},
        "cell_share_by_gap": {},
        "diffusion_spread_m_by_gap": {},
    }
    for g, label in enumerate(GAP_LABELS):
        n_g = boot.sums(block_sum_count(zs, names, cell_blocks, nb, bucket(g)))
        pooled["cell_share_by_gap"][label] = float(n_g[0] / boot.sums(count)[0])
        pooled["coverage_90_by_gap"][label] = {
            p: ci(boot.sums(block_sum(f"inside_{p}", bucket(g))) / n_g) for p in DIST
        }
        pooled["diffusion_spread_m_by_gap"][label] = float(
            boot.sums(block_sum("spread_diffusion", bucket(g)))[0] / n_g[0]
        )

    # pitch control (frames)
    pc_blocks, pc_owner = block_index(names, [zs[m]["pc_block"] for m in names])
    pboot = Boot(pc_owner)
    npb = len(pc_owner)

    def pc_sum(key):
        out = np.zeros(npb)
        for i, m in enumerate(names):
            out += np.bincount(pc_blocks[i], zs[m][f"pc_{key}"].astype(float), npb)
        return out

    n_pc = pboot.sums(np.concatenate([np.bincount(b, minlength=npb)[None] for b in pc_blocks]).sum(0).astype(float))
    agree = {v: pboot.sums(pc_sum(v)) / n_pc for v in PC}
    pooled["pitch_control"] = {
        "frames": int(n_pc[0]),
        "agreement": {v: ci(a) for v, a in agree.items()},
        "recovered_share": {
            v: ci((agree[v] - agree["visible"]) / (1 - agree["visible"])) for v in PC if v != "visible"
        },
        "diffusion_minus": {v: ci(agree["diffusion"] - agree[v]) for v in PC if v != "diffusion"},
    }

    per_match = {}
    for m in names:
        z, j = zs[m], js[m]
        sp = spearmanr(z["spread_diffusion"], z["err_diffusion"])
        per_match[m] = {
            "tune": j["tune"], "scored_cells": j["n_scored_cells"], "test_windows": j["windows"]["test"],
            "median_m": j["median_m"], "mean_m": j["mean_m"], "energy_score_m": j["energy_score_m"],
            "coverage_90": j["coverage_90"], "coverage_90_by_gap": j["coverage_90_by_gap"],
            "pitch_control_agreement": j["pitch_control_agreement"], "pitch_control_frames": j["pitch_control_frames"],
            "spread_error_spearman": float(sp.statistic),
            "transformer_best_epoch": j["transformer"]["best_epoch"], "transformer_std_scale": j["transformer"]["std_scale"],
            "diffusion_best_epoch": j["diffusion"]["best_epoch"], "diffusion_eta": j["diffusion"]["eta"],
            "gpu_seconds": j["total_seconds"],
        }  # fmt: skip

    cov_g = pooled["coverage_90_by_gap"]
    wins = sum(
        per_match[m]["median_m"]["diffusion"] < min(per_match[m]["median_m"][c] for c in CLASSICAL) for m in names
    )
    claims = {
        "C1": bool(pooled["ratio_diffusion_to_best_classical"]["position_weighted"]["ci95"][1] < 1),
        "C2": bool(pooled["ratio_diffusion_to_transformer"]["position_weighted"]["estimate"] <= 1.0),
        "C3": bool(all(0.85 <= cov_g[g]["diffusion"]["estimate"] <= 0.95 for g in GAP_LABELS)),
        "C4": bool(all(pooled["pitch_control"]["diffusion_minus"][v]["ci95"][0] > 0 for v in CLASSICAL)),
        "C5": f"{wins} of {len(names)}",
    }
    spread = [pooled["diffusion_spread_m_by_gap"][g] for g in GAP_LABELS]
    out = {
        "spec": "15-leave-one-match-out",
        "cross_validation_not_a_locked_test": True,
        "folds": names,
        "complete": len(names) == len(MATCHES),
        "fold_commits": commits,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "n_boot": N_BOOT,
        "block_minutes": 5,
        "pooled": pooled,
        "per_match": per_match,
        "spread_check": {"positive": bool(min(spread) > 0), "rises_with_gap": bool(np.all(np.diff(spread) > 0))},
        "claims": claims,
        "released_model_settings": {
            "transformer_epochs": int(np.median([per_match[m]["transformer_best_epoch"] for m in names])),
            "transformer_std_scale": float(np.median([per_match[m]["transformer_std_scale"] for m in names])),
            "diffusion_epochs": int(np.median([per_match[m]["diffusion_best_epoch"] for m in names])),
            "diffusion_eta": max(set(e := [per_match[m]["diffusion_eta"] for m in names]), key=e.count),
        },
    }
    name = "15-lomo.json" if out["complete"] else "15-lomo-partial.json"
    (Path(args.out) / name).write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {"claims": claims, "median": {p: v["estimate"] for p, v in pooled["median_m_position_weighted"].items()}},
            indent=2,
        )
    )


def block_sum_count(zs, names, cell_blocks, nb, sel) -> np.ndarray:
    out = np.zeros(nb)
    for i, m in enumerate(names):
        out += np.bincount(cell_blocks[i][sel(zs[m])], minlength=nb)
    return out


if __name__ == "__main__":
    main()
