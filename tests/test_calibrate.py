import numpy as np

from pitchvision.calibrate import cross_validate, fit, prepare, rules
from pitchvision.camera import Footprint, visible_mask
from pitchvision.tracking import Tracking

TRUE = Footprint(30.0, 60.0, 0.0)


def _synthetic(match_id, seed, true=TRUE):
    rng = np.random.default_rng(seed)
    t, n = 200, 22
    ball = np.column_stack([np.cumsum(rng.normal(0, 1, t)).clip(-40, 40), np.zeros(t)])
    players = np.stack([rng.uniform(-50, 50, (t, n)), rng.uniform(-33, 33, (t, n))], axis=-1)
    detected = visible_mask(players, ball[:, 0], true)
    corners = np.zeros((t, 4, 2))
    return Tracking(
        match_id,
        10.0,
        np.ones(t, int),
        ball,
        players,
        [str(i) for i in range(n)],
        ["home"] * 11 + ["away"] * 11,
        (105.0, 68.0),
        detected=detected,
        corners=corners,
    )


def test_fit_recovers_the_true_footprint():
    matches = [prepare(_synthetic(f"m{i}", i), score_every=1, smoothing_grid=(0.0, 0.5)) for i in range(5)]
    assert fit(matches) == (30.0, 60.0, 0.0)
    folds = cross_validate(matches, k=5)
    r = rules(folds)
    assert r["agreement"] == 1.0 and r["count_mae"] == 0.0
    assert r["C1_agreement_ge_0.85"] and r["C3_count_mae_le_1.0"]
