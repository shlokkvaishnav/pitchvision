"""Training loop for the anchored model: imputation and forecast tasks mixed 50/50, with optional early stopping."""

from __future__ import annotations

import copy

import numpy as np
import torch

from .dataset import HORIZON
from .metrics import cell_scores
from .model import gaussian_nll, predict_windows


def _t(a, device):
    return torch.as_tensor(a, dtype=torch.float32, device=device)


@torch.no_grad()
def tune_error(model, tune, device) -> float:
    """Median imputation error (m) on the hidden player cells of the tuning matches [(tr, vis, w), ...]."""
    errs = []
    for _tr, _vis, w in tune:
        hidden = ~w.visible[..., :22]
        mu, ls = predict_windows(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, device)
        errs.append(cell_scores(mu, ls, w, hidden)[0])
    return float(np.median(np.concatenate(errs)))


def _epoch(model, w, rng, opt, sched, types, device, batch: int, scaler=None) -> float:
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
        with torch.autocast("cuda", dtype=torch.float16, enabled=scaler is not None):
            mu, ls = model(p, o, types, _t(am, device), _t(al, device))
            loss = gaussian_nll(mu.float(), ls.float(), p, ~o)
        opt.zero_grad()
        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        total, n = total + loss.item() * len(idx), n + len(idx)
    return total / n


def fit(model, w, device, epochs: int, batch: int, lr: float, seed: int, tune=None, amp: bool = False):
    """AdamW + one-cycle schedule. With `tune`, keep the epoch (0 = the untrained anchor) with the lowest tune error.
    With `amp` on a CUDA device, fp16 autocast with a gradient scaler (spec 15).

    Returns (log, best_epoch). Each log entry has the epoch, the train loss and, with `tune`, the tune error.
    """
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(w.pos) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    types = torch.as_tensor(w.agent_type, device=device)
    scaler = torch.amp.GradScaler("cuda") if amp and device == "cuda" else None
    log: list[dict] = []
    best = None
    if tune is not None:
        best = (tune_error(model, tune, device), 0, copy.deepcopy(model.state_dict()))
        log.append({"epoch": 0, "train_loss": None, "tune_median_m": best[0]})
        print(f"epoch 0 tune {best[0]:.3f} m", flush=True)
    for epoch in range(1, epochs + 1):
        loss = _epoch(model, w, rng, opt, sched, types, device, batch, scaler)
        entry = {"epoch": epoch, "train_loss": loss}
        if tune is not None:
            entry["tune_median_m"] = err = tune_error(model, tune, device)
            assert best is not None
            if err < best[0]:
                best = (err, epoch, copy.deepcopy(model.state_dict()))
        log.append(entry)
        print(
            f"epoch {epoch}/{epochs} " + " ".join(f"{k} {v:.4f}" for k, v in entry.items() if k != "epoch"), flush=True
        )
    if best is None:
        return log, epochs
    model.load_state_dict(best[2])
    return log, best[1]
