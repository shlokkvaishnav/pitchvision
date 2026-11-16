"""Check that the fast loaders give the same arrays as kloppy's slow "secondspectrum" conversion.

Usage: PYTHONPATH=src python scripts/check_loader.py
"""

from __future__ import annotations

import time

import numpy as np
from kloppy import metrica, sportec

from pitchvision.tracking import from_kloppy

LIMIT = 3000


def compare(name, fast_load, ref_load):
    t = time.perf_counter()
    fast = from_kloppy(fast_load(), name)
    t_fast = time.perf_counter() - t
    t = time.perf_counter()
    ref = from_kloppy(ref_load(), name)
    t_ref = time.perf_counter() - t
    d_players = np.nanmax(np.abs(fast.players - ref.players))
    d_ball = np.nanmax(np.abs(fast.ball - ref.ball))
    same_nan = bool((np.isnan(fast.players) == np.isnan(ref.players)).all())
    print(f"{name}: max diff players {d_players:.2e} m, ball {d_ball:.2e} m, same NaN {same_nan}, "
          f"fast {t_fast:.1f} s, reference {t_ref:.1f} s")  # fmt: skip
    assert d_players < 1e-6 and d_ball < 1e-6 and same_nan


compare(
    "metrica-1",
    lambda: metrica.load_open_data(match_id=1, coordinates=None, limit=LIMIT),
    lambda: metrica.load_open_data(match_id=1, coordinates="secondspectrum", limit=LIMIT),
)
compare(
    "idsse-J03WMX",
    lambda: sportec.load_open_tracking_data("J03WMX", coordinates="sportec", limit=LIMIT),
    lambda: sportec.load_open_tracking_data("J03WMX", coordinates="secondspectrum", limit=LIMIT),
)
