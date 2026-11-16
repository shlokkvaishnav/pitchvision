"""Spec 05: masked model B with more data (IDSSE), early stopping and calibration.

Usage: PYTHONPATH=src python scripts/train_more.py --out results [--raw data/raw/skillcorner] [--epochs 30]
Runs on CPU or GPU. On Kaggle it runs from the kernel in kaggle/.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_anchored as ta

from pitchvision import baselines as bl
from pitchvision.anchor import SCALE, forecast_anchor, imputation_anchor, match_kalman
from pitchvision.model import AnchoredModel, gaussian_nll
from pitchvision.prep import (
    FPS,
    IDSSE_DEV,
    idsse_match,
    metrica_game,
    replay_mask,
    skillcorner_ids,
    skillcorner_match,
)
from pitchvision.windows import cut

HORIZON = ta.HORIZON
TRAIN = ["metrica-1", "metrica-2", "idsse-J03WMX", "idsse-J03WOY", "idsse-J03WPY"]
TUNE = ["metrica-3"]
TEST = ["idsse-J03WQQ", "idsse-J03WR9"]


def donors() -> dict[str, str]:
    ids = skillcorner_ids()
    out = {f"metrica-{g}": ids[g - 1] for g in (1, 2, 3)}
    out.update({f"idsse-{m}": ids[3 + i] for i, m in enumerate(IDSSE_DEV)})
    return out


def load_match(name: str, donor: str, raw: Path, max_windows: int = 0):
    tr = metrica_game(int(name.split("-")[1])) if name.startswith("metrica") else idsse_match(name.split("-")[1])
    vis = replay_mask(tr, skillcorner_match(donor, raw))
    w = cut(tr, vis, length=90, stride=20)
    if max_windows:
        for f in ("pos", "visible", "start", "cols"):
            setattr(w, f, getattr(w, f)[:max_windows])
    k = match_kalman(tr, vis, FPS)
    n = tr.players.shape[1]
    w.imp_mean, w.imp_ls = imputation_anchor(k, w, n)
    w.fc_mean, w.fc_ls = forecast_anchor(k, w, n, HORIZON)
    del k
    print(f"{name}: {w.pos.shape[0]} windows, donor {donor}", flush=True)
    return tr, vis, w


def concat(ws):
    out = copy.copy(ws[0])
    for f in ta.FIELDS:
        setattr(out, f, np.concatenate([getattr(w, f) for w in ws]))
    return out


def cell_errors(model, tr, vis, w, device, log_scale: float = 0.0) -> dict[str, np.ndarray]:
    """Raw errors on one match, for pooled metrics."""
    hidden = ~w.visible[..., :22]
    mu, ls = ta.predict(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, device)
    ls = ls + log_scale
    err, inside = ta._cells(mu, ls, w, hidden)
    err_anchor, inside_anchor = ta._cells(w.imp_mean, w.imp_ls, w, hidden)
    z2 = (((mu[..., :22, :] - w.pos[..., :22, :]) / np.exp(ls[..., :22, :])) ** 2)[hidden]

    kal = bl.kalman_smoother(tr.players, vis, FPS)
    frames = np.broadcast_to(w.start[:, None, None] + np.arange(w.pos.shape[1])[None, :, None], hidden.shape)
    cols = np.broadcast_to(w.cols[:, None, :], hidden.shape)
    err_kal = np.linalg.norm(kal[frames[hidden], cols[hidden]] - tr.players[frames[hidden], cols[hidden]], axis=-1)

    obs_f = w.visible.copy()
    obs_f[:, -HORIZON:, :] = False
    mu_f, _ = ta.predict(model, w.pos, obs_f, w.agent_type, w.fc_mean, w.fc_ls, device)
    err_f = np.linalg.norm((mu_f[:, -1, :22] - w.pos[:, -1, :22]) * SCALE, axis=-1).ravel()
    err_fa = np.linalg.norm((w.fc_mean[:, -1, :22] - w.pos[:, -1, :22]) * SCALE, axis=-1).ravel()
    cv = np.stack(
        [bl.constant_velocity(w.pos[b, :, :22] * SCALE, obs_f[b, :, :22], FPS)[-1] for b in range(len(w.pos))]
    )
    err_cv = np.linalg.norm(cv - w.pos[:, -1, :22] * SCALE, axis=-1).ravel()
    return {
        "err": err, "inside": inside, "err_anchor": err_anchor, "inside_anchor": inside_anchor, "z2": z2,
        "err_kal": err_kal, "err_f": err_f, "err_fa": err_fa, "err_cv": err_cv,
    }  # fmt: skip


def metrics(parts: list[dict]) -> dict:
    c = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    keep = np.isfinite(c["err_kal"])
    same = np.isfinite(c["err_cv"])
    return {
        "n_hidden_cells": int(c["err"].size),
        "imputation_median_m": float(np.median(c["err"][keep])),
        "kalman_same_cells_median_m": float(np.median(c["err_kal"][keep])),
        "anchor_imputation_median_m": float(np.median(c["err_anchor"][keep])),
        "coverage_90": float(c["inside"].mean()),
        "anchor_coverage_90": float(c["inside_anchor"].mean()),
        "forecast_4s_mean_m": float(c["err_f"][same].mean()),
        "anchor_forecast_4s_mean_m": float(c["err_fa"][same].mean()),
        "constant_velocity_4s_mean_m": float(c["err_cv"][same].mean()),
        "forecast_4s_mean_all_cells_m": float(c["err_f"].mean()),
        "m3_cell_fraction": float(same.mean()),
    }


@torch.no_grad()
def tune_error(model, tune, device) -> float:
    errs = []
    for _tr, _vis, w in tune:
        hidden = ~w.visible[..., :22]
        mu, ls = ta.predict(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, device)
        errs.append(ta._cells(mu, ls, w, hidden)[0])
    return float(np.median(np.concatenate(errs)))


def train(model, w, tune, device, epochs, batch, lr, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(w.pos) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    types = torch.as_tensor(w.agent_type, device=device)
    best = (tune_error(model, tune, device), 0, copy.deepcopy(model.state_dict()))
    log = [{"epoch": 0, "train_loss": None, "tune_median_m": best[0]}]
    print(f"epoch 0 tune {best[0]:.3f} m", flush=True)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(w.pos))
        total, n = 0.0, 0
        model.train()
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            fc = rng.random(len(idx)) < 0.5
            pos = w.pos[idx].copy()
            am = np.where(fc[:, None, None, None], w.fc_mean[idx], w.imp_mean[idx])
            al = np.where(fc[:, None, None, None], w.fc_ls[idx], w.imp_ls[idx])
            flip = rng.random(len(idx)) < 0.5
            pos[flip, ..., 0] *= -1
            am[flip, ..., 0] *= -1
            obs = w.visible[idx].copy()
            obs[fc, -HORIZON:, :] = False
            p, o = ta._t(pos, device), torch.as_tensor(obs, device=device)
            mu, ls = model(p, o, types, ta._t(am, device), ta._t(al, device))
            loss = gaussian_nll(mu, ls, p, ~o)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(idx), n + len(idx)
        err = tune_error(model, tune, device)
        log.append({"epoch": epoch, "train_loss": total / n, "tune_median_m": err})
        print(f"epoch {epoch}/{epochs} loss {total / n:.4f} tune {err:.3f} m", flush=True)
        if err < best[0]:
            best = (err, epoch, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    return log, best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--raw", default="data/raw/skillcorner")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0, help="smoke test only: limit windows per match")
    ap.add_argument("--smoke", action="store_true", help="smoke test only: Metrica games, no IDSSE")
    ap.add_argument("--commit", default=os.environ.get("PITCHVISION_COMMIT", "unknown"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw, dn = Path(args.raw), donors()
    train_ids, tune_ids, test_ids = (["metrica-1"], ["metrica-3"], ["metrica-2"]) if args.smoke else (TRAIN, TUNE, TEST)
    data = {m: load_match(m, dn[m], raw, args.max_windows) for m in train_ids + tune_ids + test_ids}
    train_w = concat([data[m][2] for m in train_ids])
    tune = [data[m] for m in tune_ids]

    model = AnchoredModel(length=90, dim=128, heads=4, depth=4).to(device)
    t0 = time.time()
    log, best_epoch = train(model, train_w, tune, device, args.epochs, args.batch, args.lr, args.seed)

    # calibration: one scale factor on the tuning match (closed form for a Gaussian)
    z2 = np.concatenate([cell_errors(model, *data[m], device)["z2"].ravel() for m in tune_ids])
    log_scale = float(0.5 * np.log(np.mean(z2)))
    tune_m = metrics([cell_errors(model, *data[m], device, log_scale) for m in tune_ids])
    test_m = metrics([cell_errors(model, *data[m], device, log_scale) for m in test_ids])
    rules = {
        "M1_ratio": test_m["imputation_median_m"] / test_m["kalman_same_cells_median_m"],
        "M1_pass": bool(test_m["imputation_median_m"] <= 0.9 * test_m["kalman_same_cells_median_m"]),
        "M2_pass": bool(0.85 <= test_m["coverage_90"] <= 0.95),
        "M3_pass": bool(test_m["forecast_4s_mean_m"] < test_m["constant_velocity_4s_mean_m"]),
    }
    out = {
        "spec": "05-more-data",
        "commit": args.commit,
        "device": device,
        "train_seconds": round(time.time() - t0, 1),
        "settings": vars(args),
        "windows": {m: len(data[m][2].pos) for m in data},
        "best_epoch": best_epoch,
        "scale_factor": float(np.exp(log_scale)),
        "log": log,
        "tuning_metrica3": tune_m,
        "test": test_m,
        "rules": rules,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "05-more-data.json").write_text(json.dumps(out, indent=2))
    torch.save(model.state_dict(), Path(args.out) / "05-more-data.pt")
    print(json.dumps({"test": test_m, "rules": rules, "best_epoch": best_epoch}, indent=2))


if __name__ == "__main__":
    main()
