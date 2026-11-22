"""Pass receiver (spec 11): match passes to tracking, build per-player features, and three models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .tracking import Tracking

VEL_LAG = 5
LANE_M = 2.0


@dataclass
class Passes:
    """P passes. x: (P, 22, 8) player features; ball: (P, 2); team: (P, 22) 1 = passer's team;
    passer, label: (P,) indices into the 22; block: (P,) str."""

    x: np.ndarray
    ball: np.ndarray
    team: np.ndarray
    passer: np.ndarray
    label: np.ndarray
    block: np.ndarray
    passer_near_ball: np.ndarray

    @staticmethod
    def concat(items: list[Passes]) -> Passes:
        f = ("x", "ball", "team", "passer", "label", "block", "passer_near_ball")
        return Passes(*(np.concatenate([getattr(p, k) for p in items]) for k in f))


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p (..., 2) to the segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def match_passes(tr: Tracking, events, match_id: str) -> Passes:
    """Completed passes with a receiver, at the nearest tracking frame (spec 11, section 2)."""
    index = {pid: i for i, pid in enumerate(tr.player_ids)}
    starts = {int(p): int(np.flatnonzero(tr.period == p)[0]) for p in np.unique(tr.period)}
    lengths = {int(p): int((tr.period == p).sum()) for p in np.unique(tr.period)}
    teams = np.array(tr.teams)
    # attack direction per (period, team): a team attacks +x if its mean x over the period is negative
    attack = {}
    for p in starts:
        rows = tr.period == p
        for g in ("home", "away"):
            attack[(p, g)] = 1.0 if np.nanmean(tr.players[rows][:, teams == g, 0]) < 0 else -1.0
    out: dict[str, list] = {k: [] for k in ("x", "ball", "team", "passer", "label", "block", "near")}
    for e in events:
        if e.event_type.name != "PASS" or e.result is None or e.result.name != "COMPLETE" or e.receiver_player is None:
            continue
        p = int(e.period.id)
        k = round(e.timestamp.total_seconds() * tr.fps)
        if p not in starts or not (VEL_LAG <= k < lengths[p]):
            continue
        t = starts[p] + k
        a, r = index.get(str(e.player.player_id)), index.get(str(e.receiver_player.player_id))
        if a is None or r is None:
            continue
        on = ~np.isnan(tr.players[t]).any(-1) & ~np.isnan(tr.players[t - VEL_LAG]).any(-1)
        idx = np.flatnonzero(on)
        if len(idx) != 22 or a not in idx or r not in idx or np.isnan(tr.ball[t]).any():
            continue
        g = teams[a]
        s = attack[(p, g)]
        pos = tr.players[t, idx] * np.array([s, s])  # flip so the passer's team attacks +x
        vel = (tr.players[t, idx] - tr.players[t - VEL_LAG, idx]) * s / (VEL_LAG / tr.fps)
        ball = tr.ball[t] * s
        same = (teams[idx] == g).astype(float)
        ia, ir = int(np.flatnonzero(idx == a)[0]), int(np.flatnonzero(idx == r)[0])
        d_pass = np.linalg.norm(pos - pos[ia], axis=-1)
        opp = pos[same == 0]
        d_opp = np.linalg.norm(pos[:, None] - opp[None], axis=-1).min(1)
        is_passer = np.zeros(22)
        is_passer[ia] = 1
        out["x"].append(np.column_stack([pos, vel, same, is_passer, d_pass, d_opp]))
        out["ball"].append(ball)
        out["team"].append(same)
        out["passer"].append(ia)
        out["label"].append(ir)
        out["block"].append(f"{match_id}-{p}-{int(k / tr.fps // 300)}")
        out["near"].append(float(np.linalg.norm(tr.players[t, a] - tr.ball[t]) <= 3.0))
    return Passes(*(np.asarray(out[k]) for k in ("x", "ball", "team", "passer", "label", "block", "near")))


def candidate_mask(p: Passes) -> np.ndarray:
    """(P, 22) True for the passer's teammates (not the passer)."""
    m = p.team == 1
    m[np.arange(len(m)), p.passer] = False
    return m


def logit_features(p: Passes) -> np.ndarray:
    """(P, 22, 4): distance to passer, distance to nearest opponent, x gain, lane blocked."""
    pos = p.x[..., :2]
    passer = pos[np.arange(len(pos)), p.passer]
    lane = np.zeros(pos.shape[:2])
    for i in range(len(pos)):
        opp = pos[i, p.team[i] == 0]
        for j in range(22):
            lane[i, j] = float((_seg_dist(opp, passer[i], pos[i, j]) <= LANE_M).any())
    return np.stack([p.x[..., 6] / 10, p.x[..., 7] / 10, (pos[..., 0] - passer[:, None, 0]) / 10, lane], -1)


class Scorer(nn.Module):
    """`logit`: a linear score per player. `gnn`: a transformer over the 22 players and the ball."""

    def __init__(self, kind: str, n_in: int):
        super().__init__()
        self.kind = kind
        if kind == "logit":
            self.lin = nn.Linear(n_in, 1, bias=False)
        else:
            self.embed = nn.Linear(n_in, 64)
            self.ball = nn.Linear(2, 64)
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 3)
            self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, ball: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.kind == "logit":
            s = self.lin(x).squeeze(-1)
        else:
            h = torch.cat([self.embed(x), self.ball(ball)[:, None]], 1)
            s = self.head(self.enc(h)[:, :22]).squeeze(-1)
        return s.masked_fill(~mask, -1e9)


def gnn_inputs(p: Passes) -> np.ndarray:
    x = p.x.copy()
    x[..., :2] /= np.array([52.5, 34.0])
    x[..., 2:4] /= 8.0
    x[..., 6:8] /= 30.0
    return x.astype(np.float32)


def fit_scorer(kind: str, x: np.ndarray, ball: np.ndarray, mask: np.ndarray, label: np.ndarray, seed: int = 0,
               epochs: int = 30, batch: int = 64, lr: float = 1e-3) -> Scorer:  # fmt: skip
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Scorer(kind, x.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    xt, bt = torch.tensor(x, dtype=torch.float32), torch.tensor(ball / np.array([52.5, 34.0]), dtype=torch.float32)
    mt, yt = torch.tensor(mask), torch.tensor(label)
    if kind == "logit":  # convex: fit to convergence with full-batch L-BFGS (no epochs or batches)
        lbfgs = torch.optim.LBFGS(m.parameters(), max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss = nn.functional.cross_entropy(m(xt, bt, mt), yt)
            loss.backward()
            return loss

        lbfgs.step(closure)
        return m
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), batch):
            b = torch.from_numpy(order[i : i + batch])
            loss = nn.functional.cross_entropy(m(xt[b], bt[b], mt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def predict_scorer(m: Scorer, x: np.ndarray, ball: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m.eval()
    s = m(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(ball / np.array([52.5, 34.0]), dtype=torch.float32),
        torch.tensor(mask),
    )
    return torch.softmax(s, -1).numpy()
