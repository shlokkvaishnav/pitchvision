"""Export the ghost-team and likely-pass overlays of the demo clips (site/demo/overlays.json).

Usage: PYTHONPATH=src .venv/bin/python scripts/export_overlays.py

For every frame of the 3 demo clips, on the positions the page draws (seen players at their camera position, hidden
players at the released model's mean, as site/app.js):
- ghost: where the spec 10 ghost model puts each player of the team without the ball;
- pass: when a player is within 2 m of the ball, the spec 11 gnn's chance for each teammate to receive.
Neither model saw J03WQQ: the ghost is trained on the other 7 development matches and the gnn on the other 4 IDSSE
development matches (as spec 13). Display only, not a result.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from downstream import GHOST_MATCHES, load_ghost, load_passes, roles_of

from pitchvision.dataset import donors, load_match
from pitchvision.ghost import Samples, predict_ghost, train_ghost
from pitchvision.passes import Passes, candidate_mask, fit_scorer, gnn_inputs, pass_features, predict_scorer
from pitchvision.prep import IDSSE_DEV

ROOT = Path(__file__).resolve().parents[1]
MATCH = "idsse-J03WQQ"
LAG = 5
CARRIER_M = 2.0


def attack_of(tr) -> dict[tuple[int, int], float]:
    """+1 if the team (0 home, 1 away) attacks +x in the period (as passes.match_passes)."""
    teams = np.array(tr.teams)
    out = {}
    for p in np.unique(tr.period):
        rows = tr.period == p
        for k, g in enumerate(("home", "away")):
            out[(int(p), k)] = 1.0 if np.nanmean(tr.players[rows][:, teams == g, 0]) < 0 else -1.0
    return out


def main() -> None:
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta = json.loads((ROOT / "site" / "demo" / "clips.json").read_text())
    tr, _vis, w = load_match(MATCH, donors()[MATCH], ROOT / "data" / "raw" / "skillcorner")
    picks = [len(w.pos) // 4, len(w.pos) // 2, 3 * len(w.pos) // 4]
    roles, attack = roles_of(tr), attack_of(tr)

    gs = Samples.concat([load_ghost(x) for x in GHOST_MATCHES if x != MATCH])
    ghost = train_ghost(gs, device)
    others = Passes.concat([load_passes(x) for x in IDSSE_DEV if f"idsse-{x}" != MATCH])
    gnn = fit_scorer("gnn", gnn_inputs(others), others.ball, candidate_mask(others), others.label)

    team = np.array(meta["agent_type"][:22])
    scale = np.array(meta["scale_m"])
    out = []
    for c, i in zip(meta["clips"], picks):
        assert abs(c["start_s"] - round(float(w.start[i] / meta["fps"]), 1)) < 1e-9, "clips.json is from another export"
        truth = np.array(c["truth_m"])  # (L, 23, 2)
        vis = np.array(c["visible"], bool)
        mean = np.array(c["python_mean"]) * scale
        pos = np.where(vis[..., None], truth, mean)[:, :22]
        ball = truth[:, 22]
        L = len(pos)
        prev = np.array([pos[max(t - LAG, 0)] for t in range(L)])
        dt = np.array([max(min(t, LAG), 1) / meta["fps"] for t in range(L)])
        prev[0] = pos[0] - (pos[1] - pos[0])  # frame 0: velocity from the next frame
        vel = (pos - prev) / dt[:, None, None]
        bprev = np.array([ball[max(t - LAG, 0)] for t in range(L)])
        bprev[0] = ball[0] - (ball[1] - ball[0])
        bvel = (ball - bprev) / dt[:, None]

        frames, holder = [], None
        rows: dict[str, list] = {"other": [], "ball": [], "role": [], "target": []}
        for t in range(L):
            f = int(w.start[i]) + t
            d = np.linalg.norm(pos[t] - ball[t], axis=-1)
            near = int(np.argmin(d))
            if d[near] <= CARRIER_M or holder is None:
                holder = near
            g = int(team[holder])  # the team on the ball; the other one is ghosted
            gs_, os_ = (slice(11, 22), slice(0, 11)) if g == 0 else (slice(0, 11), slice(11, 22))
            rows["other"].append(np.concatenate([pos[t, os_], vel[t, os_]], -1))
            rows["ball"].append(np.concatenate([ball[t], bvel[t]]))
            rows["role"].append(roles[int(tr.period[f])][w.cols[i]][gs_])
            rows["target"].append(pos[t, gs_])
            fr = {"ghost_team": 1 - g, "pass": None}
            if d[near] <= CARRIER_M:
                s = attack[(int(tr.period[f]), int(team[near]))]
                same = (team == team[near]).astype(float)
                x = pass_features(pos[t] * s, (pos[t] - vel[t] * LAG / meta["fps"]) * s, same, near, meta["fps"])
                mask = same == 1
                mask[near] = False
                p1 = Passes(x[None], (ball[t] * s)[None], same[None], np.array([near]), np.array([0]),
                            np.array(["x"]), np.zeros(1))  # fmt: skip
                prob = predict_scorer(gnn, gnn_inputs(p1), p1.ball, mask[None])[0]
                fr["pass"] = {"from": near, "p": np.round(prob, 3).tolist()}
            frames.append(fr)
        smp = Samples(*(np.asarray(rows[k]) for k in ("other", "ball", "role", "target")), np.array(["x"] * L))
        gm, _ = predict_ghost(ghost, smp, device)
        for t, fr in enumerate(frames):
            fr["ghost_m"] = np.round(gm[t], 2).tolist()
        out.append({"start_s": c["start_s"], "frames": frames})

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    doc = {
        "note": "Display only. Ghost: spec 10 model trained without J03WQQ; pass: spec 11 gnn trained without J03WQQ. "
        "ghost_m: (11, 2) metres for the team ghost_team (0 home = slots 0-10, 1 away = slots 11-21), in slot order. "
        "pass.p: (22,) chance per slot, 0 outside the passer's teammates.",
        "commit": sha.stdout.strip(),
        "clips": out,
    }
    (ROOT / "site" / "demo" / "overlays.json").write_text(json.dumps(doc, separators=(",", ":")))
    print("frames with a pass:", [sum(f["pass"] is not None for f in c["frames"]) for c in out])


if __name__ == "__main__":
    main()
