import numpy as np

from pitchvision.passes import Passes, candidate_mask, fit_scorer, gnn_inputs, logit_features, predict_scorer


def _passes(n=300, seed=0):
    """Synthetic passes: the receiver is always the teammate nearest the passer."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 22, 8))
    x[..., :2] = rng.uniform(-40, 40, (n, 22, 2))
    team = np.zeros((n, 22))
    team[:, :11] = 1
    passer = rng.integers(0, 11, n)
    for i in range(n):
        d = np.linalg.norm(x[i, :, :2] - x[i, passer[i], :2], axis=-1)
        x[i, :, 6] = d
        x[i, :, 7] = np.linalg.norm(x[i, :, None, :2] - x[i, None, 11:, :2], axis=-1).min(1)
        x[i, passer[i], 5] = 1
    x[..., 4] = team
    d = np.where(team == 1, x[..., 6], np.inf)
    d[np.arange(n), passer] = np.inf
    label = d.argmin(1)
    z = np.zeros(n)
    return Passes(x, np.zeros((n, 2)), team, passer, label, np.array(["b"] * n), z)


def test_mask_excludes_passer_and_opponents():
    p = _passes(20)
    m = candidate_mask(p)
    assert m.sum(1).tolist() == [10] * 20 and not m[np.arange(20), p.passer].any()


def test_logit_learns_nearest_teammate_and_probabilities_sum_to_one():
    tr, te = _passes(400, 0), _passes(200, 1)
    f_tr, f_te = logit_features(tr), logit_features(te)
    m = fit_scorer("logit", f_tr.astype(np.float32), tr.ball, candidate_mask(tr), tr.label, epochs=20)
    prob = predict_scorer(m, f_te.astype(np.float32), te.ball, candidate_mask(te))
    assert np.allclose(prob.sum(1), 1, atol=1e-5)
    assert (prob.argmax(1) == te.label).mean() > 0.8


def test_gnn_runs_on_inputs():
    p = _passes(64)
    m = fit_scorer("gnn", gnn_inputs(p), p.ball, candidate_mask(p), p.label, epochs=1)
    assert predict_scorer(m, gnn_inputs(p), p.ball, candidate_mask(p)).shape == (64, 22)
