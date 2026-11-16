"""Spec 04: train masked model version B (Kalman anchor) on Metrica games 1-2, validate on game 3.

Usage: PYTHONPATH=src python scripts/train_anchored.py --out results [--raw data/raw/skillcorner] [--epochs 30]
Runs on CPU or GPU. On Kaggle it runs from the kernel in kaggle/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from pitchvision import baselines as bl
from pitchvision.anchor import SCALE, forecast_anchor, imputation_anchor, match_kalman
from pitchvision.model import AnchoredModel, gaussian_nll, inside_ellipse
from pitchvision.prep import FPS, metrica_game, replay_mask, skillcorner_ids, skillcorner_match
from pitchvision.windows import cut

HORIZON = 40
FIELDS = ("pos", "visible", "start", "cols", "imp_mean", "imp_ls", "fc_mean", "fc_ls")


def load(raw: Path):
    donors = skillcorner_ids()[:3]  # spec 02 A1: first three SkillCorner IDs for games 1, 2, 3
    games = {}
    for g, donor_id in zip((1, 2, 3), donors):
        tr = metrica_game(g)
        vis = replay_mask(tr, skillcorner_match(donor_id, raw))
        w = cut(tr, vis, length=90, stride=20)
        k = match_kalman(tr, vis, FPS)
        n = tr.players.shape[1]
        w.imp_mean, w.imp_ls = imputation_anchor(k, w, n)
        w.fc_mean, w.fc_ls = forecast_anchor(k, w, n, HORIZON)
        del k
        games[g] = (tr, vis, w)
        print(f"game {g}: {w.pos.shape[0]} windows, donor {donor_id}", flush=True)
    return games


def _t(a, device, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype, device=device)


def train(model, w, device, epochs: int, batch: int, lr: float, seed: int):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(w.pos) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    types = torch.as_tensor(w.agent_type, device=device)
    log = []
    for epoch in range(epochs):
        order = rng.permutation(len(w.pos))
        total, n = 0.0, 0
        model.train()
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            fc = rng.random(len(idx)) < 0.5  # forecast task
            pos = w.pos[idx].copy()
            am = np.where(fc[:, None, None, None], w.fc_mean[idx], w.imp_mean[idx])
            al = np.where(fc[:, None, None, None], w.fc_ls[idx], w.imp_ls[idx])
            flip = rng.random(len(idx)) < 0.5
            pos[flip, ..., 0] *= -1  # flip of x; the camera side does not change
            am[flip, ..., 0] *= -1
            obs = w.visible[idx].copy()
            obs[fc, -HORIZON:, :] = False
            p, o = _t(pos, device), torch.as_tensor(obs, device=device)
            mu, ls = model(p, o, types, _t(am, device), _t(al, device))
            loss = gaussian_nll(mu, ls, p, ~o)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(idx), n + len(idx)
        log.append(total / n)
        print(f"epoch {epoch + 1}/{epochs} loss {log[-1]:.4f}", flush=True)
    return log


@torch.no_grad()
def predict(model, pos, obs, types, am, al, device, batch=64):
    model.eval()
    mus, lss = [], []
    t = torch.as_tensor(types, device=device)
    for i in range(0, len(pos), batch):
        s = slice(i, i + batch)
        mu, ls = model(
            _t(pos[s], device), torch.as_tensor(obs[s], device=device), t, _t(am[s], device), _t(al[s], device)
        )
        mus.append(mu.cpu().numpy())
        lss.append(ls.cpu().numpy())
    return np.concatenate(mus), np.concatenate(lss)


def _cells(mu, ls, w, hidden):
    err = np.linalg.norm((mu[..., :22, :] - w.pos[..., :22, :]) * SCALE, axis=-1)[hidden]
    inside = inside_ellipse(
        torch.as_tensor(mu[..., :22, :]), torch.as_tensor(ls[..., :22, :]), torch.as_tensor(w.pos[..., :22, :])
    ).numpy()[hidden]
    return err, inside


def evaluate(model, tr, vis, w, device) -> dict:
    hidden = ~w.visible[..., :22]
    mu, ls = predict(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, device)
    err_model, inside = _cells(mu, ls, w, hidden)
    err_anchor, inside_anchor = _cells(w.imp_mean, w.imp_ls, w, hidden)

    # The bar baseline of spec 02 on the full match, scored on the same cells.
    kal = bl.kalman_smoother(tr.players, vis, FPS)
    frames = np.broadcast_to(w.start[:, None, None] + np.arange(w.pos.shape[1])[None, :, None], hidden.shape)
    cols = np.broadcast_to(w.cols[:, None, :], hidden.shape)
    err_kal = np.linalg.norm(kal[frames[hidden], cols[hidden]] - tr.players[frames[hidden], cols[hidden]], axis=-1)

    obs_f = w.visible.copy()
    obs_f[:, -HORIZON:, :] = False
    mu_f, _ = predict(model, w.pos, obs_f, w.agent_type, w.fc_mean, w.fc_ls, device)
    err_f = np.linalg.norm((mu_f[:, -1, :22] - w.pos[:, -1, :22]) * SCALE, axis=-1).ravel()
    err_fa = np.linalg.norm((w.fc_mean[:, -1, :22] - w.pos[:, -1, :22]) * SCALE, axis=-1).ravel()
    cv = np.stack(
        [bl.constant_velocity(w.pos[b, :, :22] * SCALE, obs_f[b, :, :22], FPS)[-1] for b in range(len(w.pos))]
    )
    err_cv = np.linalg.norm(cv - w.pos[:, -1, :22] * SCALE, axis=-1).ravel()

    same = np.isfinite(err_cv)  # spec 04 A1: M3 on the cells where the baseline has an estimate
    keep = np.isfinite(err_kal)
    return {
        "n_windows": len(w.pos),
        "n_hidden_cells": int(hidden.sum()),
        "imputation_median_m": float(np.median(err_model[keep])),
        "kalman_same_cells_median_m": float(np.median(err_kal[keep])),
        "anchor_imputation_median_m": float(np.median(err_anchor[keep])),
        "coverage_90": float(inside.mean()),
        "anchor_coverage_90": float(inside_anchor.mean()),
        "forecast_4s_mean_m": float(err_f[same].mean()),
        "anchor_forecast_4s_mean_m": float(err_fa[same].mean()),
        "constant_velocity_4s_mean_m": float(err_cv[same].mean()),
        "forecast_4s_mean_all_cells_m": float(err_f.mean()),
        "m3_cell_fraction": float(same.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--raw", default="data/raw/skillcorner")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0, help="smoke test only: limit train and validation windows")
    ap.add_argument("--commit", default=os.environ.get("PITCHVISION_COMMIT", "unknown"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    games = load(Path(args.raw))
    train_w, w2 = games[1][2], games[2][2]
    for f in FIELDS:
        setattr(train_w, f, np.concatenate([getattr(train_w, f), getattr(w2, f)]))
    if args.max_windows:
        for w in (train_w, games[3][2]):
            for f in FIELDS:
                setattr(w, f, getattr(w, f)[: args.max_windows])

    model = AnchoredModel(length=90, dim=128, heads=4, depth=4).to(device)
    t0 = time.time()
    log = train(model, train_w, device, args.epochs, args.batch, args.lr, args.seed)
    tr3, vis3, w3 = games[3]
    m = evaluate(model, tr3, vis3, w3, device)
    rules = {
        "M1_ratio": m["imputation_median_m"] / m["kalman_same_cells_median_m"],
        "M1_pass": bool(m["imputation_median_m"] <= 0.9 * m["kalman_same_cells_median_m"]),
        "M2_pass": bool(0.85 <= m["coverage_90"] <= 0.95),
        "M3_pass": bool(m["forecast_4s_mean_m"] < m["constant_velocity_4s_mean_m"]),
    }
    out = {
        "spec": "04-anchored-model",
        "commit": args.commit,
        "device": device,
        "train_seconds": round(time.time() - t0, 1),
        "settings": vars(args),
        "train_windows": len(train_w.pos),
        "loss_per_epoch": log,
        "validation_game3": m,
        "rules": rules,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "04-anchored-model.json").write_text(json.dumps(out, indent=2))
    torch.save(model.state_dict(), Path(args.out) / "04-anchored-model.pt")
    print(json.dumps({"validation_game3": m, "rules": rules}, indent=2))


if __name__ == "__main__":
    main()
