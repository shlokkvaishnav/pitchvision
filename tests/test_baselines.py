import numpy as np
import pytest

from pitchvision.baselines import (
    constant_velocity,
    errors,
    formation_template,
    hold_last,
    kalman_smoother,
    linear_interpolation,
)

FPS = 10.0


def _straight_line(t=50, n=2, speed=5.0):
    """Players move along x at a constant speed. Player j starts at y = 10 j."""
    time = np.arange(t) / FPS
    x = np.repeat((speed * time)[:, None], n, axis=1)
    y = np.repeat((10.0 * np.arange(n))[None], t, axis=0)
    return np.stack([x, y], axis=-1)


def _gap_mask(t=50, n=2, start=20, stop=30):
    vis = np.ones((t, n), dtype=bool)
    vis[start:stop, 0] = False
    return vis


def test_hold_last_keeps_the_last_seen_position():
    pos, vis = _straight_line(), _gap_mask()
    est = hold_last(pos, vis)
    assert np.allclose(est[25, 0], pos[19, 0])
    assert np.allclose(est[:, 1], pos[:, 1])


def test_linear_and_constant_velocity_are_exact_on_a_straight_line():
    pos, vis = _straight_line(), _gap_mask()
    hidden = ~vis
    assert errors(linear_interpolation(pos, vis), pos, hidden)["mean"] == pytest.approx(0.0, abs=1e-9)
    assert errors(constant_velocity(pos, vis, FPS), pos, hidden)["mean"] == pytest.approx(0.0, abs=1e-9)


def test_constant_velocity_caps_the_speed():
    pos, vis = _straight_line(speed=20.0), _gap_mask()
    est = constant_velocity(pos, vis, FPS, max_speed=8.0)
    step = np.linalg.norm(est[21, 0] - est[20, 0]) * FPS
    assert step == pytest.approx(8.0)


def test_kalman_smoother_is_close_on_a_straight_line():
    pos, vis = _straight_line(), _gap_mask()
    e = errors(kalman_smoother(pos, vis, FPS), pos, ~vis)
    assert e["mean"] < 0.5


def test_formation_template_uses_teammate_centroid():
    t = 40
    base = _straight_line(t=t, n=3)
    vis = np.ones((t, 3), dtype=bool)
    vis[10:20, 0] = False
    est, offsets = formation_template(base, vis, ["home", "home", "home"])
    # all players move together, so centroid + mean offset is close to the truth
    assert errors(est, base, ~vis)["mean"] < 1.0
    assert offsets.shape == (3, 2)


def test_errors_ignore_missing_truth():
    pos = np.array([[[0.0, 0.0]], [[np.nan, np.nan]]])
    est = np.array([[[3.0, 4.0]], [[0.0, 0.0]]])
    e = errors(est, pos, np.ones((2, 1), dtype=bool))
    assert e["n"] == 1 and e["median"] == pytest.approx(5.0)
