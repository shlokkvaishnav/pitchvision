"""Conditional diffusion over hidden player positions (docs/specs/15-leave-one-match-out.md, section 4).

CSDI / U2Diff recipe on the axial backbone: the denoiser sees the noisy residual (hidden cells only), the observed
positions and flags, and the Kalman anchor; it predicts v = sqrt(abar) noise - sqrt(1 - abar) x0 (spec 15, amendment A1).
The residual x0 is (truth - anchor mean) in units of 5 m.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from .anchor import SCALE
from .model import AxialBlock, time_embedding

STEPS = 50
BETA_START, BETA_END = 1e-4, 0.5
RESIDUAL_M = 5.0  # metres per residual unit
SAMPLE_STEPS = 10
K_SAMPLES = 20
X0_CLIP = 25.0  # residual units: 125 m, more than the pitch; keeps early DDIM steps finite
_RES_SCALE = torch.as_tensor(SCALE / RESIDUAL_M, dtype=torch.float32)  # normalised units -> residual units


def schedule(steps: int = STEPS) -> torch.Tensor:
    """alpha_bar for each step, CSDI's quadratic beta schedule."""
    beta = torch.linspace(BETA_START**0.5, BETA_END**0.5, steps, dtype=torch.float64) ** 2
    return torch.cumprod(1 - beta, 0).float()


def step_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    arg = t.float()[:, None] * freq[None]
    return torch.cat([torch.sin(arg), torch.cos(arg)], -1)


class Denoiser(nn.Module):
    time: torch.Tensor

    def __init__(self, length: int = 90, dim: int = 128, heads: int = 4, depth: int = 4):
        super().__init__()
        self.dim = dim
        self.inp = nn.Linear(9, dim)  # noisy residual (2), observed position (2), observed flag, anchor mean, log-std
        self.agent_type = nn.Embedding(3, dim)
        self.register_buffer("time", time_embedding(length, dim), persistent=False)
        self.step = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(AxialBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, noisy, pos, observed, agent_type, anchor_mean, anchor_log_std, t):
        """noisy, pos, anchors: (B, L, A, 2); observed: (B, L, A) bool; t: (B,) step index. Returns v (B, L, A, 2)."""
        obs = observed.unsqueeze(-1).to(noisy.dtype)
        x = torch.cat([noisy * (1 - obs), torch.nan_to_num(pos) * obs, obs, anchor_mean, anchor_log_std], -1)
        h = self.inp(x) + self.agent_type(agent_type)[None, None] + self.time[: pos.shape[1]][None, :, None]
        h = h + self.step(step_embedding(t, self.dim))[:, None, None]
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


def residual(pos: torch.Tensor, anchor_mean: torch.Tensor) -> torch.Tensor:
    return (torch.nan_to_num(pos) - anchor_mean) * _RES_SCALE.to(pos.device)


def to_positions(res: torch.Tensor, anchor_mean: torch.Tensor) -> torch.Tensor:
    return anchor_mean + res / _RES_SCALE.to(res.device)


def loss_fn(model, pos, observed, types, am, al, abar, gen=None) -> torch.Tensor:
    """v-prediction MSE on hidden player cells (the ball is always observed)."""
    b = pos.shape[0]
    t = torch.randint(0, len(abar), (b,), device=pos.device, generator=gen)
    noise = torch.randn(pos.shape, device=pos.device, generator=gen)
    a = abar.to(pos.device)[t].view(b, 1, 1, 1)
    x0 = residual(pos, am)
    noisy = a.sqrt() * x0 + (1 - a).sqrt() * noise
    pred = model(noisy, pos, observed, types, am, al, t)
    target = a.sqrt() * noise - (1 - a).sqrt() * x0
    w = (~observed).unsqueeze(-1).float()
    return ((pred.float() - target) ** 2 * w).sum() / w.sum().clamp(min=1.0) / 2


def _t(a, device):
    return torch.as_tensor(a, dtype=torch.float32, device=device)


@torch.no_grad()
def tune_loss(model, w, device, abar, batch: int = 64) -> float:
    """Denoising loss on a tuning match with fixed noise and steps (seed 0)."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    types = torch.as_tensor(w.agent_type, device=device)
    total, n = 0.0, 0
    for i in range(0, len(w.pos), batch):
        s = slice(i, i + batch)
        obs = torch.as_tensor(w.visible[s], device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            loss = loss_fn(model, _t(w.pos[s], device), obs, types, _t(w.imp_mean[s], device),
                           _t(w.imp_ls[s], device), abar, gen)  # fmt: skip
        total, n = total + float(loss) * len(w.pos[s]), n + len(w.pos[s])
    return total / n


def fit(model, w, device, epochs: int, batch: int, lr: float, seed: int, tune=None, amp: bool = True):
    """AdamW + one-cycle. With `tune` (a Windows), keep the epoch with the lowest tuning denoising loss."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    abar = schedule()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(w.pos) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    types = torch.as_tensor(w.agent_type, device=device)
    log: list[dict] = []
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(w.pos))
        total, n = 0.0, 0
        for i in range(0, len(order), batch):
            idx = np.sort(order[i : i + batch])
            pos, am = w.pos[idx].copy(), w.imp_mean[idx].copy()
            flip = rng.random(len(idx)) < 0.5
            pos[flip, ..., 0] *= -1
            am[flip, ..., 0] *= -1
            obs = torch.as_tensor(w.visible[idx], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model, _t(pos, device), obs, types, _t(am, device), _t(w.imp_ls[idx], device), abar, gen)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(idx), n + len(idx)
        entry = {"epoch": epoch, "train_loss": total / n}
        if tune is not None:
            entry["tune_loss"] = tl = tune_loss(model, tune, device, abar)
            if best is None or tl < best[0]:
                best = (tl, epoch, copy.deepcopy(model.state_dict()))
        log.append(entry)
        print("diffusion " + " ".join(f"{k} {v:.4f}" for k, v in entry.items()), flush=True)
    if best is None:
        return log, epochs
    model.load_state_dict(best[2])
    return log, best[1]


@torch.no_grad()
def sample(model, pos, observed, types, am, al, k: int = K_SAMPLES, steps: int = SAMPLE_STEPS, eta: float = 0.0,
           seed: int = 0, batch: int = 32, device: str = "cpu") -> np.ndarray:  # fmt: skip
    """DDIM samples of all positions, normalised units, (B, K, L, A, 2). Observed cells keep their true value."""
    model.eval()
    abar = schedule().to(device)
    ts = torch.linspace(len(abar) - 1, 0, steps).round().long().tolist()
    gen = torch.Generator(device=device).manual_seed(seed)
    ty = torch.as_tensor(types, device=device)
    out = []
    for i in range(0, len(pos), batch):
        s = slice(i, i + batch)
        b = len(pos[s])
        p = _t(pos[s], device).repeat_interleave(k, 0)
        o = torch.as_tensor(observed[s], device=device).repeat_interleave(k, 0)
        m = _t(am[s], device).repeat_interleave(k, 0)
        lsd = _t(al[s], device).repeat_interleave(k, 0)
        x = torch.randn(p.shape, device=device, generator=gen)
        for j, t in enumerate(ts):
            a = abar[t]
            a_prev = abar[ts[j + 1]] if j + 1 < len(ts) else torch.tensor(1.0, device=device)
            tt = torch.full((len(p),), t, device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                v = model(x, p, o, ty, m, lsd, tt).float()
            x0 = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-X0_CLIP, X0_CLIP)
            eps = (1 - a).sqrt() * x + a.sqrt() * v
            sigma = eta * ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).clamp(min=0).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps
            if j + 1 < len(ts):
                x = x + sigma * torch.randn(p.shape, device=device, generator=gen)
        est = to_positions(x, m)
        est = torch.where(o.unsqueeze(-1), torch.nan_to_num(p), est)
        out.append(est.view(b, k, *est.shape[1:]).cpu().numpy())
    return np.concatenate(out)


def region_threshold(k: int = K_SAMPLES, level: float = 0.9) -> float:
    """Threshold on (y - m)' S^-1 (y - m) for the Gaussian 90% prediction region from k samples (spec 15, section 4)."""
    from scipy.stats import f

    return (k + 1) / k * 2 * (k - 1) / (k - 2) * float(f.ppf(level, 2, k - 2))


def sample_stats(samples_m: np.ndarray, truth_m: np.ndarray) -> dict[str, np.ndarray]:
    """Per cell scores from samples (N, K, 2) in metres and truth (N, 2): mean/median error, energy score, inside, spread."""
    k = samples_m.shape[1]
    mean = samples_m.mean(1)
    med = np.median(samples_m, axis=1)
    d = samples_m - mean[:, None]
    cov = np.einsum("nki,nkj->nij", d, d) / (k - 1) + np.eye(2) * 1e-6
    r = truth_m - mean
    q = np.einsum("ni,nij,nj->n", r, np.linalg.inv(cov), r) * k / (k + 1)
    return {
        "err_mean": np.linalg.norm(r, axis=-1),
        "err_median": np.linalg.norm(truth_m - med, axis=-1),
        "es": energy_score(samples_m, truth_m),
        "inside": q <= region_threshold(k),
        "spread": np.linalg.norm(d, axis=-1).mean(1),
    }


def energy_score(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Energy score per cell: mean ||X_k - y|| - 1/2 mean over k != j of ||X_k - X_j||. samples (N, K, 2), truth (N, 2)."""
    k = samples.shape[1]
    term1 = np.linalg.norm(samples - truth[:, None], axis=-1).mean(1)
    pair = np.linalg.norm(samples[:, :, None] - samples[:, None], axis=-1).sum((1, 2)) / (k * (k - 1))
    return term1 - 0.5 * pair
