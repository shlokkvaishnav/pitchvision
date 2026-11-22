"""Spec 11: pass receiver, leave-one-match-out over the 5 IDSSE development matches (exploratory).

Usage: PYTHONPATH=src .venv/bin/python scripts/pass_receiver.py
Writes results/11-pass-receiver.json and data/models/pass_gnn.pt (trained on all 5 matches, for the site).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from pitchvision.passes import (
    Passes,
    candidate_mask,
    fit_scorer,
    gnn_inputs,
    logit_features,
    match_passes,
    predict_scorer,  # fmt: skip
)
from pitchvision.prep import IDSSE_DEV, idsse_match

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "passes"


def load(m: str) -> Passes:
    path = CACHE / f"{m}.npz"
    if path.exists():
        z = np.load(path)
        return Passes(*(z[k] for k in ("x", "ball", "team", "passer", "label", "block", "passer_near_ball")))
    from kloppy import sportec

    ev = sportec.load_open_event_data(match_id=m)
    p = match_passes(idsse_match(m), ev.events, f"idsse-{m}")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=p.x, ball=p.ball, team=p.team, passer=p.passer, label=p.label, block=p.block,
                        passer_near_ball=p.passer_near_ball)  # fmt: skip
    return p


def nll(prob: np.ndarray, label: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(prob[np.arange(len(label)), label], 1e-9, 1))


def main() -> None:
    torch.set_num_threads(4)
    data = {m: load(m) for m in IDSSE_DEV}
    for m, p in data.items():
        print(
            m, len(p), "passes, passer within 3 m of the ball:", round(float(p.passer_near_ball.mean()), 3), flush=True
        )
    res: dict[str, list] = {k: [] for k in ("nearest", "logit", "gnn")}
    labels, blocks = [], []
    for m in IDSSE_DEV:
        tr = Passes.concat([data[x] for x in IDSSE_DEV if x != m])
        te = data[m]
        mk_tr, mk_te = candidate_mask(tr), candidate_mask(te)
        near = np.where(mk_te, -te.x[..., 6], -np.inf)
        p_near = np.exp(near - near.max(1, keepdims=True))
        res["nearest"].append(p_near / p_near.sum(1, keepdims=True))
        lg = fit_scorer("logit", logit_features(tr).astype(np.float32), tr.ball, mk_tr, tr.label)
        res["logit"].append(predict_scorer(lg, logit_features(te).astype(np.float32), te.ball, mk_te))
        gn = fit_scorer("gnn", gnn_inputs(tr), tr.ball, mk_tr, tr.label)
        res["gnn"].append(predict_scorer(gn, gnn_inputs(te), te.ball, mk_te))
        labels.append(te.label)
        blocks.append(te.block)
        print(m, {k: round(float((v[-1].argmax(1) == te.label).mean()), 3) for k, v in res.items()}, flush=True)
    y, bl = np.concatenate(labels), np.concatenate(blocks)
    prob = {k: np.concatenate(v) for k, v in res.items()}
    summary = {}
    for k, pr in prob.items():
        top3 = np.argsort(-pr, 1)[:, :3]
        summary[k] = {
            "top1": float((pr.argmax(1) == y).mean()),
            "top3": float((top3 == y[:, None]).any(1).mean()),
            "log_loss": float(nll(pr, y).mean()),
        }
    d = nll(prob["gnn"], y) - nll(prob["logit"], y)
    keys, inv = np.unique(bl, return_inverse=True)
    s, c = np.bincount(inv, d, len(keys)), np.bincount(inv, minlength=len(keys))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), (10_000, len(keys)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    rules = {
        "P1_diff_gnn_minus_logit": float(d.mean()),
        "P1_ci": ci,
        "P1_pass": bool(d.mean() < 0 and ci[1] < 0),
        "P2_pass": summary["gnn"]["top1"] >= summary["nearest"]["top1"],
    }
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    out = {
        "spec": "11-pass-receiver",
        "commit": sha,
        "exploratory": "leave-one-match-out over 5 IDSSE development matches; holdout not loaded",
        "passes": len(y),
        "passer_within_3m_of_ball": float(np.mean(np.concatenate([p.passer_near_ball for p in data.values()]))),
        "models": summary,
        "rules": rules,
    }
    (ROOT / "results" / "11-pass-receiver.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("passes", "passer_within_3m_of_ball", "models", "rules")}, indent=2))
    allp = Passes.concat(list(data.values()))
    final = fit_scorer("gnn", gnn_inputs(allp), allp.ball, candidate_mask(allp), allp.label)
    (ROOT / "data" / "models").mkdir(parents=True, exist_ok=True)
    torch.save(final.state_dict(), ROOT / "data" / "models" / "pass_gnn.pt")


if __name__ == "__main__":
    main()
