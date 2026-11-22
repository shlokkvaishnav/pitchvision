import numpy as np
import torch

from pitchvision.ghost import GhostModel, Samples, fit_linear_shift, predict_linear_shift, samples
from pitchvision.tracking import Tracking


def _tracking(t=40, sub_off=True):
    rng = np.random.default_rng(0)
    n = 23 if sub_off else 22
    pos = rng.uniform(-40, 40, (t, n, 2)) * np.array([1, 0.6])
    pos = pos + np.linspace(0, 3, t)[:, None, None]
    if sub_off:
        pos[:, 22] = np.nan  # a substitute who never plays
    ball = rng.uniform(-30, 30, (t, 2))
    teams = ["home"] * 11 + ["away"] * 11 + (["home"] if sub_off else [])
    return Tracking("m", 10.0, np.ones(t, int), ball, pos, [str(i) for i in range(n)], teams, (105.0, 68.0))


def test_samples_two_per_frame_and_skip_the_first_lag_frames():
    s = samples(_tracking())
    assert len(s) == 2 * (40 - 5)
    assert s.other.shape[1:] == (11, 4) and s.role.shape[1:] == (11, 2) and s.target.shape[1:] == (11, 2)


def test_frames_without_eleven_a_side_are_left_out():
    tr = _tracking()
    tr.players[20:, 3] = np.nan  # a home player sent off at frame 20
    assert len(samples(tr)) == 2 * (20 - 5)


def test_linear_shift_recovers_a_linear_rule():
    s = samples(_tracking())
    s.target = s.role + 0.2 * s.ball[:, None, :2]
    w = fit_linear_shift(s)
    assert np.allclose(predict_linear_shift(w, s), s.target, atol=1e-6)


def test_ghost_model_starts_from_the_role_shape():
    m = GhostModel(dim=32, heads=2, depth=1)
    out, log_std = m(torch.zeros(3, 11, 4), torch.zeros(3, 4), torch.ones(3, 11, 2))
    assert out.shape == (3, 11, 2) and log_std.shape == (3, 11, 2)
    assert Samples.concat([samples(_tracking())] * 2).ball.shape[0] == 2 * 70
