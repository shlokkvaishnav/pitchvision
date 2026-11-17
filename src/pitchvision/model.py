"""Masked model, version A: factorized transformer with a Gaussian output (docs/specs/03-masked-model.md)."""

from __future__ import annotations

import math

import torch
from torch import nn

CHI2_2DOF_90 = 4.605170185988091  # 90% quantile of the chi-square distribution with 2 degrees of freedom


def time_embedding(length: int, dim: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    emb = torch.zeros(length, dim)
    emb[:, 0::2] = torch.sin(pos * freq)
    emb[:, 1::2] = torch.cos(pos * freq)
    return emb


class AxialBlock(nn.Module):
    """Attention over time for each agent, then attention over agents for each frame, then an MLP."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.attn_a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, length, a, d = h.shape
        x = h.permute(0, 2, 1, 3).reshape(b * a, length, d)  # time attention
        y = self.norm_t(x)
        x = x + self.attn_t(y, y, y, need_weights=False)[0]
        x = x.reshape(b, a, length, d).permute(0, 2, 1, 3).reshape(b * length, a, d)  # agent attention
        y = self.norm_a(x)
        x = x + self.attn_a(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.norm_m(x))
        return x.reshape(b, length, a, d)


class MaskedModel(nn.Module):
    time: torch.Tensor  # registered buffer (sinusoidal time embedding)

    def __init__(self, length: int = 90, dim: int = 128, heads: int = 4, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Linear(3, dim)  # x, y (zero if hidden), observed flag
        self.agent_type = nn.Embedding(3, dim)
        self.register_buffer("time", time_embedding(length, dim), persistent=False)
        self.blocks = nn.ModuleList(AxialBlock(dim, heads, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 4)

    def forward(self, pos: torch.Tensor, observed: torch.Tensor, agent_type: torch.Tensor):
        """pos: (B, L, A, 2), observed: (B, L, A) bool, agent_type: (A,). Returns mean and log-std, (B, L, A, 2)."""
        obs = observed.unsqueeze(-1).float()
        x = torch.cat([torch.nan_to_num(pos) * obs, obs], dim=-1)
        h = self.inp(x) + self.agent_type(agent_type)[None, None] + self.time[: pos.shape[1]][None, :, None]
        for block in self.blocks:
            h = block(h)
        out = self.head(self.norm(h))
        return out[..., :2], out[..., 2:].clamp(-7.0, 3.0)


def gaussian_nll(mean, log_std, target, weight) -> torch.Tensor:
    """Mean negative log-likelihood of a diagonal Gaussian on the cells where weight is True."""
    z = (torch.nan_to_num(target) - mean) * torch.exp(-log_std)
    nll = 0.5 * z.pow(2) + log_std
    w = weight.unsqueeze(-1).float()
    return (nll * w).sum() / w.sum().clamp(min=1.0) / 2


def inside_ellipse(mean, log_std, target, scale=(1.0, 1.0), q: float = CHI2_2DOF_90) -> torch.Tensor:
    """True where the target is inside the Gaussian ellipse with the given chi-square quantile."""
    s = torch.tensor(scale, dtype=mean.dtype, device=mean.device)
    z = ((target - mean) * s) / (torch.exp(log_std) * s)
    return z.pow(2).sum(-1) <= q


class AnchoredModel(MaskedModel):
    """Version B (docs/specs/04-anchored-model.md): the output is a correction to a Kalman anchor."""

    def __init__(self, length: int = 90, dim: int = 128, heads: int = 4, depth: int = 4, dropout: float = 0.0):
        super().__init__(length, dim, heads, depth, dropout)
        self.inp = nn.Linear(7, dim)  # x, y (zero if hidden), observed flag, anchor x, y, anchor log-std x, y
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pos, observed, agent_type, anchor_mean, anchor_log_std):
        obs = observed.unsqueeze(-1).float()
        x = torch.cat([torch.nan_to_num(pos) * obs, obs, anchor_mean, anchor_log_std], dim=-1)
        h = self.inp(x) + self.agent_type(agent_type)[None, None] + self.time[: pos.shape[1]][None, :, None]
        for block in self.blocks:
            h = block(h)
        out = self.head(self.norm(h))
        return anchor_mean + out[..., :2], (anchor_log_std + out[..., 2:]).clamp(-7.0, 3.0)


@torch.no_grad()
def predict_windows(model, pos, obs, types, anchor_mean, anchor_log_std, device, batch: int = 64):
    """Run an AnchoredModel over windows in batches. Arrays in, arrays out: (mean, log_std), normalised units."""
    import numpy as np

    def t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    model.eval()
    mus, lss = [], []
    ty = torch.as_tensor(types, device=device)
    for i in range(0, len(pos), batch):
        s = slice(i, i + batch)
        mu, ls = model(t(pos[s]), torch.as_tensor(obs[s], device=device), ty, t(anchor_mean[s]), t(anchor_log_std[s]))
        mus.append(mu.cpu().numpy())
        lss.append(ls.cpu().numpy())
    return np.concatenate(mus), np.concatenate(lss)
