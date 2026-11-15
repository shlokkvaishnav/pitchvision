import numpy as np
import torch

from pitchvision.model import MaskedModel, gaussian_nll, inside_ellipse
from pitchvision.tracking import Tracking
from pitchvision.windows import BALL, cut, task_mask


def _match(t=120, n=24):
    rng = np.random.default_rng(0)
    players = np.stack([rng.uniform(-50, 50, (t, n)), rng.uniform(-33, 33, (t, n))], axis=-1)
    players[:, 23] = np.nan  # a substitute who is not on the pitch
    teams = ["home"] * 12 + ["away"] * 12
    return Tracking(
        "m",
        10.0,
        np.ones(t, int),
        np.zeros((t, 2)),
        players,
        [str(i) for i in range(n)],
        teams,
        (105.0, 68.0),
    )


def test_cut_needs_11_per_team():
    tr = _match()
    vis = np.ones((120, 24), dtype=bool)
    w = cut(tr, vis, length=90, stride=10)
    assert w.pos.shape[0] == 0  # 12 home players present: not a valid window
    tr.players[:, 11] = np.nan
    w = cut(tr, vis, length=90, stride=10)
    assert w.pos.shape == (4, 90, 23, 2)
    assert w.agent_type[-1] == BALL and w.visible[..., -1].all()
    assert np.abs(w.pos).max() <= 1.0


def test_task_mask_hides_the_future():
    vis = np.ones((2, 90, 23), dtype=bool)
    obs = task_mask(vis, np.array([True, False]), horizon=40)
    assert not obs[0, 50:].any() and obs[0, :50].all() and obs[1].all()


def test_model_is_permutation_equivariant_over_players():
    torch.manual_seed(0)
    m = MaskedModel(length=20, dim=32, heads=4, depth=2).eval()
    pos = torch.randn(1, 20, 23, 2)
    obs = torch.rand(1, 20, 23) > 0.3
    types = torch.tensor([0] * 11 + [1] * 11 + [2])
    perm = torch.cat([torch.randperm(11), 11 + torch.randperm(11), torch.tensor([22])])
    mu, _ = m(pos, obs, types)
    mu_p, _ = m(pos[:, :, perm], obs[:, :, perm], types[perm])
    assert torch.allclose(mu[:, :, perm], mu_p, atol=1e-5)


def test_loss_decreases_on_a_tiny_batch():
    torch.manual_seed(0)
    m = MaskedModel(length=20, dim=32, heads=4, depth=2)
    pos = torch.randn(4, 20, 23, 2) * 0.3
    obs = torch.rand(4, 20, 23) > 0.5
    types = torch.tensor([0] * 11 + [1] * 11 + [2])
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        mu, ls = m(pos, obs, types)
        loss = gaussian_nll(mu, ls, pos, ~obs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5


def test_inside_ellipse():
    mean = torch.zeros(3, 2)
    log_std = torch.zeros(3, 2)
    target = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 1.2]])
    assert inside_ellipse(mean, log_std, target).tolist() == [True, True, False]


def test_windows_remember_source_cells():
    tr = _match()
    tr.players[:, 11] = np.nan
    w = cut(tr, np.ones((120, 24), dtype=bool), length=90, stride=10)
    b, slot, t = 2, 5, 7
    frame, col = w.start[b] + t, w.cols[b, slot]
    assert np.allclose(w.pos[b, t, slot] * [52.5, 34.0], tr.players[frame, col])
