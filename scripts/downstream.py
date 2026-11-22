"""Spec 13: what filling in hidden players gives back to pitch control, pass prediction and the ghost team.

Usage: PYTHONPATH=src .venv/bin/python scripts/downstream.py [--raw data/raw/skillcorner]
Writes results/13-downstream.json. Exploratory: IDSSE J03WQQ and J03WR9 only (spec 05 test matches, never trained on).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from kloppy import sportec

from pitchvision.anchor import SCALE
from pitchvision.dataset import donors, load_match
from pitchvision.ghost import Samples, predict_ghost, train_ghost
from pitchvision.model import AnchoredModel, predict_windows
from pitchvision.passes import (
    Passes,
    candidate_mask,
    fit_scorer,
    gnn_inputs,
    match_passes,
    pass_features,
    predict_scorer,
)
from pitchvision.pitch_control import control_map
from pitchvision.prep import IDSSE_DEV

ROOT = Path(__file__).resolve().parents[1]
EVAL = ["J03WQQ", "J03WR9"]
MID, LAG = 45, 5
VERSIONS = ("truth", "visible", "kalman", "model")
GHOST_MATCHES = [f"metrica-{g}" for g in (1, 2, 3)] + [f"idsse-{m}" for m in IDSSE_DEV]


def load_passes(m: str) -> Passes:
    z = np.load(ROOT / "data" / "passes" / f"{m}.npz")
    return Passes(*(z[k] for k in ("x", "ball", "team", "passer", "label", "block", "passer_near_ball")))


def load_ghost(name: str) -> Samples:
    z = np.load(ROOT / "data" / "ghost" / f"{name}.npz")
    return Samples(z["other"], z["ball"], z["role"], z["target"], z["block"])


def roles_of(tr) -> dict[int, np.ndarray]:
    """Mean position of every tracking column in each period (the spec 10 role)."""
    on = ~np.isnan(tr.players).any(-1)
    out = {}
    for p in np.unique(tr.period):
        rows = tr.period == p
        with np.errstate(all="ignore"):
            out[int(p)] = np.nanmean(np.where(on[rows][..., None], tr.players[rows], np.nan), axis=0)
    return out


def block_of(m: str, tr, frame: int) -> str:
    return f"{m}-{int(tr.period[frame])}-{int(frame / tr.fps // 300)}"


def boot_ratio(blocks, num, den, n=10_000, seed=0) -> list[float]:
    keys, inv = np.unique(blocks, return_inverse=True)
    sn, sd = np.bincount(inv, num, len(keys)), np.bincount(inv, den, len(keys))
    idx = np.random.default_rng(seed).integers(0, len(keys), (n, len(keys)))
    r = sn[idx].sum(1) / sd[idx].sum(1)
    return [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]


def boot_mean(blocks, x, n=10_000, seed=0) -> list[float]:
    return boot_ratio(blocks, x, np.ones_like(x), n, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "skillcorner"))
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AnchoredModel(length=90, dim=128, heads=4, depth=4)
    model.load_state_dict(torch.load(ROOT / "data" / "models" / "05-more-data.pt", map_location="cpu"))
    model.to(device)

    A = {v: [] for v in VERSIONS}
    A_blocks: list[str] = []
    B = {v: [] for v in ("truth", "kalman", "model")}
    B_hidden, B_blocks = [], []
    C = {v: [] for v in ("truth", "kalman", "model")}
    C_blocks: list[str] = []
    for m in EVAL:
        name = f"idsse-{m}"
        tr, _vis, w = load_match(name, donors()[name], Path(args.raw))
        mu, _ = predict_windows(model, w.pos, w.visible, w.agent_type, w.imp_mean, w.imp_ls, device)
        seen = w.visible[..., :22]
        truth = w.pos[..., :22, :] * SCALE
        series = {
            "truth": truth,
            "kalman": np.where(seen[..., None], truth, w.imp_mean[..., :22, :] * SCALE),
            "model": np.where(seen[..., None], truth, mu[..., :22, :] * SCALE),
        }
        ball = w.pos[..., 22, :] * SCALE
        team = np.r_[np.zeros(11, int), np.ones(11, int)]

        # A: pitch control at the middle frame of each window with a hidden player
        for b in range(len(w.pos)):
            hid = ~seen[b, MID]
            if not hid.any():
                continue
            maps = {}
            for v in ("truth", "kalman", "model"):
                pos, vel = series[v][b, MID], (series[v][b, MID] - series[v][b, MID - LAG]) / (LAG / tr.fps)
                maps[v] = control_map(pos, vel, team, ball[b, MID]) >= 0.5
            k = ~hid
            vt = (truth[b, MID] - truth[b, MID - LAG]) / (LAG / tr.fps)
            maps["visible"] = control_map(truth[b, MID][k], vt[k], team[k], ball[b, MID]) >= 0.5
            for v in VERSIONS:
                A[v].append(float((maps[v] == maps["truth"]).mean()))
            A_blocks.append(block_of(m, tr, int(w.start[b]) + MID))
        print(m, "A units", len(A_blocks), flush=True)

        # B: pass receiver, gnn trained on the other 4 IDSSE development matches
        others = Passes.concat([load_passes(x) for x in IDSSE_DEV if x != m])
        torch.set_num_threads(4)
        gnn = fit_scorer("gnn", gnn_inputs(others), others.ball, candidate_mask(others), others.label)
        _passes, frames, players, attack = match_passes(tr, sportec.load_open_event_data(match_id=m).events, name, True)
        centres = w.start + MID
        for t, (a, r) in zip(frames, players):
            b = int(np.argmin(np.abs(centres - t)))
            j = t - int(w.start[b])
            if abs(int(centres[b]) - t) > 10 or not (LAG <= j < w.pos.shape[1]):
                continue
            cols = list(w.cols[b])
            if a not in cols or r not in cols:
                continue
            ia, ir = cols.index(a), cols.index(r)
            side = "home" if ia < 11 else "away"
            s = attack[(int(tr.period[t]), side)]
            same = (team == team[ia]).astype(float)
            mask = same == 1
            mask[ia] = False
            for v in ("truth", "kalman", "model"):
                x = pass_features(series[v][b, j] * s, series[v][b, j - LAG] * s, same, ia, tr.fps)[None]
                p1 = Passes(
                    x, (ball[b, j] * s)[None], same[None], np.array([ia]), np.array([ir]), np.array(["x"]), np.zeros(1)
                )
                prob = predict_scorer(
                    gnn,
                    gnn_inputs(p1),
                    p1.ball,
                    mask[None],
                )
                B[v].append(float(prob.argmax(1)[0] == ir))
            B_hidden.append(float(not seen[b, j, ir]))
            B_blocks.append(block_of(m, tr, t))
        print(m, "B passes", len(B_blocks), flush=True)

        # C: ghost team, model trained on the other 7 development matches (spec 10 settings)
        gs = Samples.concat([load_ghost(x) for x in GHOST_MATCHES if x != name])
        ghost = train_ghost(gs, device)
        roles = roles_of(tr)
        rows = {v: {"other": [], "ball": [], "role": [], "target": []} for v in C}
        for b in range(len(w.pos)):
            if not (~seen[b, MID]).any():
                continue
            f = int(w.start[b]) + MID
            r_now = roles[int(tr.period[f])][w.cols[b]]
            bvel = (ball[b, MID] - ball[b, MID - LAG]) / (LAG / tr.fps)
            for g_slots, o_slots in ((slice(0, 11), slice(11, 22)), (slice(11, 22), slice(0, 11))):
                for v in C:
                    pos = series[v][b, MID, o_slots]
                    vel = (pos - series[v][b, MID - LAG, o_slots]) / (LAG / tr.fps)
                    rows[v]["other"].append(np.concatenate([pos, vel], -1))
                    rows[v]["ball"].append(np.concatenate([ball[b, MID], bvel]))
                    rows[v]["role"].append(r_now[g_slots])
                    rows[v]["target"].append(truth[b, MID, g_slots])
                C_blocks.append(block_of(m, tr, f))
        for v, errs in C.items():
            smp = Samples(
                *(np.asarray(rows[v][k]) for k in ("other", "ball", "role", "target")),
                np.array(C_blocks[-len(rows[v]["ball"]) :]),
            )
            mean, _ = predict_ghost(ghost, smp, device)
            errs.extend(np.linalg.norm(mean - smp.target, axis=-1).mean(1).tolist())
        print(m, "C samples", len(C_blocks), flush=True)

    out: dict = {"matches": EVAL, "model": "data/models/05-more-data.pt", "exploratory": True}
    a = {v: np.array(x) for v, x in A.items()}
    ab = np.array(A_blocks)
    lost = 1 - a["visible"]
    out["A_pitch_control"] = {
        "units": len(ab),
        **{f"agreement_{v}": float(a[v].mean()) for v in VERSIONS},
        **{f"recovered_share_{v}": float((a[v] - a["visible"]).sum() / lost.sum()) for v in ("kalman", "model")},
        **{f"recovered_share_{v}_ci": boot_ratio(ab, a[v] - a["visible"], lost) for v in ("kalman", "model")},
    }
    bb = np.array(B_blocks)
    out["B_pass_receiver"] = {
        "passes": len(bb),
        **{f"top1_{v}": float(np.mean(x)) for v, x in B.items()},
        **{f"top1_{v}_ci": boot_mean(bb, np.array(x)) for v, x in B.items()},
        "receiver_off_camera_share": float(np.mean(B_hidden)),
    }
    cb = np.array(C_blocks)
    out["C_ghost"] = {
        "samples": len(cb),
        **{f"mean_error_{v}_m": float(np.mean(x)) for v, x in C.items()},
        **{f"gap_to_truth_{v}_ci": boot_mean(cb, np.array(C[v]) - np.array(C["truth"])) for v in ("kalman", "model")},
    }
    out["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    (ROOT / "results" / "13-downstream.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
