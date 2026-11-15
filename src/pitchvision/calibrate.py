"""Fit the trapezoid camera mask on SkillCorner broadcast data (see specs/01-data-and-camera.md, section 4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Footprint, camera_centre, polygon_mask
from .tracking import Tracking

NEAR_GRID = np.arange(20.0, 70.0 + 1e-9, 5.0)
FAR_GRID = np.arange(40.0, 110.0 + 1e-9, 5.0)
SMOOTHING_GRID = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98)


@dataclass
class Prepared:
    """One match, reduced to the frames that we score."""

    match_id: str
    centre: dict[float, np.ndarray]  # smoothing -> (F,) camera centre before the clip at the goal lines
    x: np.ndarray  # (F, N) player position along x
    half_length: float
    depth: np.ndarray  # (F, N) 0 = near touchline, 1 = far touchline
    valid: np.ndarray  # (F, N) player has a position
    detected: np.ndarray  # (F, N)
    reference: np.ndarray  # (F, N) mask from the real corners


def prepare(tr: Tracking, score_every: int, smoothing_grid=SMOOTHING_GRID) -> Prepared:
    if tr.detected is None or tr.corners is None:
        raise ValueError("calibration needs broadcast data with detected flags and corners")
    keep = np.isin(tr.period, (1, 2))
    tr = tr.select(np.flatnonzero(keep))
    rows = np.arange(0, len(tr.period), score_every)
    length, width = tr.pitch
    x, y = tr.players[rows, :, 0], tr.players[rows, :, 1]
    # A 1 m footprint gives an almost unclipped centre. _mask applies the clip for each far_width.
    centre = {a: camera_centre(tr.ball[:, 0], Footprint(1.0, 1.0, a), length)[rows] for a in smoothing_grid}
    valid = np.isfinite(x) & np.isfinite(y)
    return Prepared(
        match_id=tr.match_id,
        centre=centre,
        x=x,
        half_length=length / 2,
        depth=np.clip((y + width / 2) / width, 0.0, 1.0),
        valid=valid,
        detected=tr.detected[rows] & valid,
        reference=polygon_mask(tr.players[rows], tr.corners[rows]),
    )


def _mask(m: Prepared, near: float, far: float, a: float) -> np.ndarray:
    limit = max(m.half_length - far / 2, 0.0)
    centre = np.clip(m.centre[a], -limit, limit)[:, None]
    half = (near + m.depth * (far - near)) / 2
    with np.errstate(invalid="ignore"):
        return (np.abs(m.x - centre) <= half) & m.valid


def scores(m: Prepared, mask: np.ndarray) -> dict[str, float]:
    agree = (mask == m.detected)[m.valid]
    count_diff = np.abs(mask.sum(1) - m.detected.sum(1))
    return {"agreement": float(agree.mean()), "count_mae": float(count_diff.mean()), "n": int(m.valid.sum())}


def fit(matches: list[Prepared]) -> tuple[float, float, float]:
    """Return (near_width, far_width, pan_smoothing) with the highest agreement, pooled over player-frames."""
    best, best_hits = None, -1
    for a in matches[0].centre:
        for near in NEAR_GRID:
            for far in FAR_GRID[FAR_GRID >= near]:
                hits = sum(int(((_mask(m, near, far, a) == m.detected) & m.valid).sum()) for m in matches)
                if hits > best_hits:
                    best, best_hits = (float(near), float(far), float(a)), hits
    return best


def evaluate(m: Prepared, params: tuple[float, float, float]) -> dict:
    fitted = scores(m, _mask(m, *params))
    ref = scores(m, m.reference)
    return {"match_id": m.match_id, "fitted": fitted, "reference": ref}


def cross_validate(matches: list[Prepared], k: int = 5) -> list[dict]:
    """k-fold cross-validation by match. Returns one record per fold."""
    order = sorted(matches, key=lambda m: m.match_id)
    folds = [order[i::k] for i in range(k)]
    out = []
    for i, test in enumerate(folds):
        train = [m for j, f in enumerate(folds) if j != i for m in f]
        params = fit(train)
        per_match = [evaluate(m, params) for m in test]
        w = np.array([r["fitted"]["n"] for r in per_match], dtype=float)
        out.append(
            {
                "fold": i,
                "params": {"near_width": params[0], "far_width": params[1], "pan_smoothing": params[2]},
                "test_matches": [m.match_id for m in test],
                "agreement": float(np.average([r["fitted"]["agreement"] for r in per_match], weights=w)),
                "reference_agreement": float(np.average([r["reference"]["agreement"] for r in per_match], weights=w)),
                "count_mae": float(np.mean([r["fitted"]["count_mae"] for r in per_match])),
                "per_match": per_match,
            }
        )
    return out


def rules(folds: list[dict]) -> dict:
    agreement = float(np.mean([f["agreement"] for f in folds]))
    reference = float(np.mean([f["reference_agreement"] for f in folds]))
    count = float(np.mean([f["count_mae"] for f in folds]))
    return {
        "agreement": agreement,
        "reference_agreement": reference,
        "count_mae": count,
        "C1_agreement_ge_0.85": agreement >= 0.85,
        "C2_within_5pp_of_reference": agreement >= reference - 0.05,
        "C3_count_mae_le_1.0": count <= 1.0,
    }
