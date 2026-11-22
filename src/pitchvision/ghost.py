"""Ghost team (spec 10): where would each player of one team stand, given the other team, the ball and their role?

Coordinates are metres, centred (x right, y up). A sample is one frame with one team ghosted; each frame gives two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .tracking import Tracking

SCALE = np.array([52.5, 34.0])
VEL_LAG = 5  # frames (0.5 s at 10 Hz)
CHI2_2DOF_90 = 4.605170185988091


@dataclass
class Samples:
    """Arrays for S samples. other: (S, 11, 4) pos+vel; ball: (S, 4); role, target: (S, 11, 2); block: (S,) str."""

    other: np.ndarray
    ball: np.ndarray
    role: np.ndarray
    target: np.ndarray
    block: np.ndarray

    def __len__(self) -> int:
        return len(self.ball)

    @staticmethod
    def concat(items: list[Samples]) -> Samples:
        return Samples(
            *(np.concatenate([getattr(s, f) for s in items]) for f in ("other", "ball", "role", "target", "block"))
        )


def samples(tr: Tracking) -> Samples:
    """All frames with 11 v 11 on the pitch, ball known, and the same 22 players 0.5 s earlier in the same period."""
    pos, ball, per = tr.players, tr.ball, tr.period
    teams = np.array(tr.teams)
    on = ~np.isnan(pos).any(-1)  # (T, N)
    roles = np.full((len(np.unique(per)), pos.shape[1], 2), np.nan)
    periods = list(np.unique(per))
    for i, p in enumerate(periods):
        with np.errstate(all="ignore"):
            roles[i] = np.nanmean(np.where(on[per == p, :, None], pos[per == p], np.nan), axis=0)
    out: dict[str, list] = {k: [] for k in ("other", "ball", "role", "target", "block")}
    for t in range(VEL_LAG, len(per)):
        if per[t] != per[t - VEL_LAG] or np.isnan(ball[t]).any() or np.isnan(ball[t - VEL_LAG]).any():
            continue
        idx = {g: np.flatnonzero(on[t] & (teams == g)) for g in ("home", "away")}
        if any(len(v) != 11 for v in idx.values()):
            continue
        both = np.concatenate(list(idx.values()))
        if not on[t - VEL_LAG, both].all():
            continue
        vel = (pos[t] - pos[t - VEL_LAG]) / (VEL_LAG / tr.fps)
        bvel = (ball[t] - ball[t - VEL_LAG]) / (VEL_LAG / tr.fps)
        r = roles[periods.index(per[t])]
        block = f"{tr.match_id}-{per[t]}-{int(t / tr.fps // 300)}"
        for g, o in (("home", "away"), ("away", "home")):
            out["other"].append(np.concatenate([pos[t, idx[o]], vel[idx[o]]], -1))
            out["ball"].append(np.concatenate([ball[t], bvel]))
            out["role"].append(r[idx[g]])
            out["target"].append(pos[t, idx[g]])
            out["block"].append(block)
    return Samples(*(np.asarray(out[k]) for k in ("other", "ball", "role", "target", "block")))


# ---------- baselines ----------
def _linear_features(s: Samples) -> np.ndarray:
    """(S, 11, F): ball pos and vel, role, other team's centroid, and a constant."""
    n = len(s)
    ball = np.repeat(s.ball[:, None, :], 11, 1) / np.tile(SCALE, 2)
    cen = np.repeat(s.other[:, :, :2].mean(1)[:, None, :], 11, 1) / SCALE
    return np.concatenate([ball, s.role / SCALE, cen, np.ones((n, 11, 1))], -1)


def fit_linear_shift(s: Samples) -> np.ndarray:
    """Least squares for (target - role) from the linear features, pooled over all players. Returns (F, 2)."""
    x = _linear_features(s).reshape(-1, 9)
    y = (s.target - s.role).reshape(-1, 2)
    return np.linalg.lstsq(x, y, rcond=None)[0]


def predict_linear_shift(w: np.ndarray, s: Samples) -> np.ndarray:
    return s.role + _linear_features(s) @ w


# ---------- model ----------
class GhostModel(nn.Module):
    """Transformer encoder over 23 tokens: 11 other-team players, the ball, 11 ghost queries (their role).

    Each ghost token outputs an offset from its role (mean) and a log std, in normalised units.
    """

    def __init__(self, dim: int = 128, heads: int = 4, depth: int = 4):
        super().__init__()
        self.other = nn.Linear(4, dim)
        self.ball = nn.Linear(4, dim)
        self.ghost = nn.Linear(2, dim)
        self.kind = nn.Parameter(torch.zeros(3, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, 4)

    def forward(self, other: torch.Tensor, ball: torch.Tensor, role: torch.Tensor):
        h = torch.cat(
            [
                self.other(other) + self.kind[0],
                self.ball(ball)[:, None] + self.kind[1],
                self.ghost(role) + self.kind[2],
            ],
            1,
        )
        out = self.head(self.enc(h)[:, 12:])
        return role + out[..., :2], out[..., 2:].clamp(-6, 3)


def to_tensors(s: Samples, device) -> tuple[torch.Tensor, ...]:
    sc = torch.tensor(np.tile(SCALE, 2), dtype=torch.float32)
    f = lambda a, d: torch.tensor(a, dtype=torch.float32) / d
    return tuple(t.to(device) for t in (f(s.other, sc), f(s.ball, sc), f(s.role, sc[:2]), f(s.target, sc[:2])))


def train_ghost(s: Samples, device, epochs: int = 8, batch: int = 512, lr: float = 3e-4, seed: int = 0) -> GhostModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GhostModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    other, ball, role, target = to_tensors(s, device)
    for _ in range(epochs):
        order = torch.tensor(rng.permutation(len(s)), device=device)
        for i in range(0, len(s), batch):
            b = order[i : i + batch]
            mean, log_std = model(other[b], ball[b], role[b])
            loss = (log_std + 0.5 * ((target[b] - mean) / log_std.exp()) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_ghost(model: GhostModel, s: Samples, device, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Mean (S, 11, 2) in metres and std (S, 11, 2) in metres."""
    model.eval()
    other, ball, role, _ = to_tensors(s, device)
    means, stds = [], []
    for i in range(0, len(s), batch):
        m, ls = model(other[i : i + batch], ball[i : i + batch], role[i : i + batch])
        means.append(m.cpu().numpy() * SCALE)
        stds.append(ls.exp().cpu().numpy() * SCALE)
    return np.concatenate(means), np.concatenate(stds)


def inside_90(mean: np.ndarray, std: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (((target - mean) / std) ** 2).sum(-1) <= CHI2_2DOF_90
