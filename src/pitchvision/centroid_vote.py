"""Role-anchored centroid voting (Choi 2026, arXiv 2607.11548, baseline B4), reimplemented for 10 Hz tracking.

Online and training-free. For each team and frame: every visible player with a stored role offset votes for the team
centre as (position - offset); with at least 3 votes the centre is the mean vote, else the visible mean. Offsets are an
EMA of (position - centre) over visible frames. A hidden player is placed at centre + offset, falling back to the last
visible-centre offset and then to the last seen position. Choi's EMA weight is 0.1 per 0.2 s step; at 10 Hz the weight
per frame is 1 - 0.9 ** 0.5, the same time constant (docs/specs/15-leave-one-match-out.md, section 3).
"""

from __future__ import annotations

import numpy as np

EMA_WEIGHT = 1 - 0.9**0.5
MIN_VOTES = 3
MIN_VISIBLE = 3


def _team(pos: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """One team, one period. pos (T, n, 2) with truth, vis (T, n). Returns the estimate (T, n, 2), NaN if no guess."""
    t, n, _ = pos.shape
    out = np.full((t, n, 2), np.nan)
    ema = np.full((n, 2), np.nan)
    snap = np.full((n, 2), np.nan)
    last = np.full((n, 2), np.nan)
    for k in range(t):
        v = vis[k] & np.isfinite(pos[k]).all(-1)
        p = pos[k]
        cen = vote = None
        if v.sum() >= MIN_VISIBLE:
            cen = p[v].mean(0)
            snap[v] = p[v] - cen
            voters = v & np.isfinite(ema).all(-1)
            if voters.sum() >= MIN_VOTES:
                vote = (p[voters] - ema[voters]).mean(0)
            ref = vote if vote is not None else cen
            off = p[v] - ref
            old = np.where(np.isfinite(ema[v]), ema[v], off)
            ema[v] = (1 - EMA_WEIGHT) * old + EMA_WEIGHT * off
            voters = v & np.isfinite(ema).all(-1)  # Choi votes again with the updated offsets to impute
            vote = (p[voters] - ema[voters]).mean(0) if voters.sum() >= MIN_VOTES else None
        last[v] = p[v]
        out[k, v] = p[v]
        hid = ~vis[k]
        if not hid.any():
            continue
        est = last.copy()
        if cen is not None:
            has_snap = np.isfinite(snap).all(-1)
            est[has_snap] = cen + snap[has_snap]
        if vote is not None:
            has_ema = np.isfinite(ema).all(-1)
            est[has_ema] = vote + ema[has_ema]
        out[k, hid] = est[hid]
    return out


def centroid_vote(positions: np.ndarray, visible: np.ndarray, teams: list[str]) -> np.ndarray:
    """Estimate (T, N, 2) for one period. Visible cells keep their value."""
    out = np.full(positions.shape, np.nan)
    side = np.asarray(teams)
    for team in ("home", "away"):
        cols = np.flatnonzero(side == team)
        out[:, cols] = _team(positions[:, cols], visible[:, cols])
    return out


def centroid_vote_by_period(positions, visible, teams, period) -> np.ndarray:
    out = np.full(positions.shape, np.nan)
    for p in np.unique(period):
        rows = period == p
        out[rows] = centroid_vote(positions[rows], visible[rows], teams)
    return out
