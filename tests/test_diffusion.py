import numpy as np
import torch

from pitchvision.centroid_vote import centroid_vote
from pitchvision.diffusion import Denoiser, energy_score, loss_fn, region_threshold, sample, sample_stats, schedule


def _batch(b=2, length=12, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-0.8, 0.8, (b, length, 23, 2)).astype(np.float32)
    vis = rng.random((b, length, 23)) < 0.5
    vis[..., 22] = True
    am = (pos + rng.normal(0, 0.05, pos.shape)).astype(np.float32)
    al = np.full(pos.shape, -3.0, np.float32)
    types = np.array([0] * 11 + [1] * 11 + [2])
    return pos, vis, am, al, types


def test_untrained_denoiser_predicts_zero_and_loss_is_finite():
    pos, vis, am, al, types = _batch()
    m = Denoiser(length=12, dim=32, heads=2, depth=1)
    t = torch.as_tensor
    out = m(t(pos), t(pos), t(vis), t(types), t(am), t(al), torch.zeros(2, dtype=torch.long))
    assert torch.all(out == 0)
    loss = loss_fn(m, t(pos), t(vis), t(types), t(am), t(al), schedule())
    assert torch.isfinite(loss) and 0.0 < loss.item() < 1.5


def test_samples_keep_observed_cells_and_vary_on_hidden_cells():
    pos, vis, am, al, types = _batch()
    m = Denoiser(length=12, dim=32, heads=2, depth=1)
    s = sample(m, pos, vis, types, am, al, k=4, steps=3, eta=1.0)
    assert s.shape == (2, 4, 12, 23, 2)
    np.testing.assert_allclose(s[:, 0][vis], pos[vis], atol=1e-6)
    assert s[:, :, ~vis[0]].std() > 0 or s[1][:, ~vis[1]].std() > 0


def test_energy_score_of_a_point_is_its_error_and_region_threshold_is_above_chi2():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(5, 2))
    pt = np.repeat((y + 1.0)[:, None], 20, 1)
    np.testing.assert_allclose(energy_score(pt, y), np.sqrt(2.0))
    assert 4.605 < region_threshold(20) < 6.5


def test_gaussian_samples_give_about_90_percent_coverage():
    rng = np.random.default_rng(1)
    n = 20_000
    samples = rng.normal(size=(n, 20, 2)) * [2.0, 0.5]
    truth = rng.normal(size=(n, 2)) * [2.0, 0.5]
    cov = sample_stats(samples, truth)["inside"].mean()
    assert 0.885 < cov < 0.915


def test_centroid_vote_follows_a_team_that_moves_together():
    t, n = 50, 11
    base = np.stack([np.linspace(-30, 30, n), np.zeros(n)], -1)
    shift = np.stack([np.linspace(0, 10, t), np.zeros(t)], -1)
    pos = base[None] + shift[:, None]
    vis = np.ones((t, n), bool)
    vis[25:, 0] = False  # player 0 hidden for the second half
    est = centroid_vote(pos, vis, ["home"] * n)
    err = np.linalg.norm(est[25:, 0] - pos[25:, 0], axis=-1)
    hold = np.linalg.norm(pos[24, 0] - pos[25:, 0], axis=-1)
    assert err.mean() < 0.3 * hold.mean()
