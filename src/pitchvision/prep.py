"""Shared data preparation: Metrica games with the replay_ball mask (spec 02, amendment A1)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

from .camera import footprint_sequences, polygon_mask, replay
from .splits import assert_not_holdout
from .tracking import (
    SKILLCORNER_MEDIA,
    SKILLCORNER_RAW,
    Tracking,
    broadcast_fields,
    from_kloppy,
    load_skillcorner_cached,
    resample,
)

MATCHES_URL = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches.json"
FPS = 10.0


def skillcorner_ids() -> list[str]:
    with urllib.request.urlopen(MATCHES_URL) as r:
        return sorted(str(m["id"]) for m in json.load(r))


def skillcorner_match(match_id: str, raw_dir: Path) -> Tracking:
    folder = Path(raw_dir) / match_id
    folder.mkdir(parents=True, exist_ok=True)
    tracking, match = folder / "tracking.jsonl", folder / "match.json"
    if not (folder / "parsed.npz").exists():
        if not tracking.exists() or tracking.stat().st_size < 10_000:
            urllib.request.urlretrieve(
                f"{SKILLCORNER_MEDIA}/{match_id}/{match_id}_tracking_extrapolated.jsonl", tracking
            )
        if not match.exists():
            urllib.request.urlretrieve(f"{SKILLCORNER_RAW}/{match_id}/{match_id}_match.json", match)
    return load_skillcorner_cached(folder)


def replay_corners(tr: Tracking, donor: Tracking) -> np.ndarray:
    """The camera footprint (T, 4, 2) of the replay_ball mask with a SkillCorner donor (periods 1 and 2)."""
    _, corners = broadcast_fields(donor)
    keep = np.isin(donor.period, (1, 2))
    seqs = footprint_sequences(donor.period[keep], donor.ball[keep, 0], corners[keep])
    return replay(seqs, tr.period, tr.ball[:, 0])


def replay_mask(tr: Tracking, donor: Tracking) -> np.ndarray:
    """The replay_ball mask with a SkillCorner donor match (periods 1 and 2 of the donor)."""
    return polygon_mask(tr.players, replay_corners(tr, donor))


def metrica_game(game: int) -> Tracking:
    from kloppy import metrica

    match_id = f"metrica-{game}"
    assert_not_holdout(match_id)
    ds = metrica.load_open_data(match_id=game, coordinates=None)  # normalised: fast path of from_kloppy
    return resample(from_kloppy(ds, match_id), FPS)


IDSSE_DEV = ("J03WMX", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")  # spec 05: the development matches, sorted


def idsse_match(match: str, *, allow_holdout: bool = False) -> Tracking:
    """One IDSSE match (Bassek et al. 2025, CC BY 4.0), resampled to 10 Hz.

    Refuses holdout matches. Only the final spec (07) sets allow_holdout.
    """
    from kloppy import sportec

    match_id = f"idsse-{match}"
    assert_not_holdout(match_id, allow_holdout=allow_holdout)
    ds = sportec.load_open_tracking_data(match, coordinates="sportec")  # native: centred metres
    return resample(from_kloppy(ds, match_id), FPS)
