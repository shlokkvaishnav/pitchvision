"""Match loading for training and evaluation: tracking, camera mask, windows and Kalman anchors."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .anchor import forecast_anchor, imputation_anchor, match_kalman
from .prep import FPS, IDSSE_DEV, idsse_match, metrica_game, replay_mask, skillcorner_ids, skillcorner_match
from .windows import Windows, cut

HORIZON = 40  # forecast frames (4.0 s at 10 Hz)
FIELDS = ("pos", "visible", "start", "cols", "imp_mean", "imp_ls", "fc_mean", "fc_ls")


def donors() -> dict[str, str]:
    """Camera-mask donor (a SkillCorner match ID) for each development match (specs 02 A1 and 05)."""
    ids = skillcorner_ids()
    out = {f"metrica-{g}": ids[g - 1] for g in (1, 2, 3)}
    out.update({f"idsse-{m}": ids[3 + i] for i, m in enumerate(IDSSE_DEV)})
    return out


def load_match(name: str, donor: str, raw: Path, max_windows: int = 0, *, allow_holdout: bool = False):
    """Tracking, camera mask and windows (with imputation and forecast anchors) for `metrica-<g>` or `idsse-<id>`."""
    if name.startswith("metrica"):
        tr = metrica_game(int(name.split("-")[1]))
    else:
        tr = idsse_match(name.split("-")[1], allow_holdout=allow_holdout)
    vis = replay_mask(tr, skillcorner_match(donor, raw))
    w = cut(tr, vis, length=90, stride=20)
    if max_windows:
        for f in ("pos", "visible", "start", "cols"):
            setattr(w, f, getattr(w, f)[:max_windows])
    k = match_kalman(tr, vis, FPS)
    n = tr.players.shape[1]
    w.imp_mean, w.imp_ls = imputation_anchor(k, w, n)
    w.fc_mean, w.fc_ls = forecast_anchor(k, w, n, HORIZON)
    del k
    print(f"{name}: {w.pos.shape[0]} windows, donor {donor}", flush=True)
    return tr, vis, w


def concat(ws: list[Windows]) -> Windows:
    """Join the windows of several matches."""
    out = copy.copy(ws[0])
    for f in FIELDS:
        setattr(out, f, np.concatenate([getattr(w, f) for w in ws]))
    return out
