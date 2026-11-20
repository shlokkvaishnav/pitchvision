"""Export the in-browser demo: the anchored model as ONNX and three clips of IDSSE match J03WQQ.

Usage: PYTHONPATH=src python scripts/export_demo.py --weights data/models/05-more-data.pt --tag spec05 --std-scale 1.1747
Writes site/demo/model.onnx, site/demo/clips.json and tests/js/kalman-ref.json.

The page runs the Kalman smoother itself (site/kalman.js). Each clip carries the observed positions (mm) of its 23 tracks
over a context around the window: back and forward until every track has CONTEXT_SEEN observed frames, at most
CONTEXT_MAX frames on each side. Hidden positions are not in the context. tests/js/kalman-ref.json holds the Python
smoother output on the same context, for the JS parity test.

Clips: the windows at 1/4, 1/2 and 3/4 of the match (a fixed rule, no selection by result).
J03WQQ is a test match of spec 05, not a holdout match. IDSSE data: Bassek et al. (2025), CC BY 4.0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from pitchvision.anchor import CLIP, MIN_STD_M, SCALE, _normalise
from pitchvision.baselines import kalman
from pitchvision.dataset import donors, load_match
from pitchvision.model import AnchoredModel
from pitchvision.prep import FPS, replay_corners, skillcorner_match

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "demo"
MATCH = "idsse-J03WQQ"
CONTEXT_SEEN = 50  # observed frames of each track before and after the window
CONTEXT_MAX = 3000  # frames on each side, at most (5 min)


def context_bounds(seen: np.ndarray, start: int, stop: int) -> tuple[int, int]:
    """First and last+1 frame of the context of window [start, stop). seen: (T, 23) bool."""
    a, b = start, stop
    for j in range(seen.shape[1]):
        before, after = np.flatnonzero(seen[:start, j]), np.flatnonzero(seen[stop:, j]) + stop
        a = min(a, before[-CONTEXT_SEEN] if before.size >= CONTEXT_SEEN else 0)
        b = max(b, after[CONTEXT_SEEN - 1] + 1 if after.size >= CONTEXT_SEEN else len(seen))
    return max(a, start - CONTEXT_MAX), min(b, stop + CONTEXT_MAX)


def encode_tracks(pos_m: np.ndarray, seen: np.ndarray) -> list[list[list[int]]]:
    """(T, 23, 2) metres and (T, 23) observed -> for each track, its observed runs [first frame, x0, y0, x1, y1, ...] in mm."""
    mm = np.round(np.nan_to_num(pos_m) * 1000).astype(int)
    out = []
    for j in range(mm.shape[1]):
        edges = np.flatnonzero(np.diff(np.concatenate([[0], seen[:, j].astype(int), [0]])))
        out.append([[int(s0), *mm[s0:s1, j].ravel().tolist()] for s0, s1 in zip(edges[::2], edges[1::2])])
    return out


def decode_tracks(tracks, frames: int) -> np.ndarray:
    """Inverse of encode_tracks, in metres: (T, 23, 2) float64, NaN where not observed (as site/demo-core.js)."""
    out = np.full((frames, len(tracks), 2), np.nan)
    for j, runs in enumerate(tracks):
        for run in runs:
            xy = np.asarray(run[1:], float).reshape(-1, 2) / 1000
            out[run[0] : run[0] + len(xy), j] = xy
    return out


class OnnxWrapper(torch.nn.Module):
    """ONNX has no bool input in onnxruntime-web for all backends; the observed mask is a float (1 = seen)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pos, observed, agent_type, anchor_mean, anchor_log_std):
        return self.model(pos, observed > 0.5, agent_type, anchor_mean, anchor_log_std)


def r(a, n=3):
    return np.round(np.nan_to_num(np.asarray(a, float), nan=0.0), n).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--raw", default="data/raw/skillcorner")
    ap.add_argument("--std-scale", type=float, default=1.0, help="calibration factor of the spec (05: results JSON)")
    args = ap.parse_args()

    model = AnchoredModel(length=90, dim=128, heads=4, depth=4)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    tr, vis_mask, w = load_match(MATCH, donors()[MATCH], Path(args.raw))
    camera = replay_corners(tr, skillcorner_match(donors()[MATCH], Path(args.raw)))  # (T, 4, 2) metres
    picks = [len(w.pos) // 4, len(w.pos) // 2, 3 * len(w.pos) // 4]
    types = torch.as_tensor(w.agent_type)
    n = tr.players.shape[1]
    tracks = np.concatenate([tr.players, tr.ball[:, None]], axis=1)  # (T, N+1, 2), as pitchvision.anchor
    seen_all = np.concatenate([vis_mask, np.ones((len(vis_mask), 1), bool)], axis=1) & np.isfinite(tracks).all(-1)
    clips, refs, gaps = [], [], []
    length = w.pos.shape[1]
    for i in picks:
        s0 = int(w.start[i])
        cols = np.concatenate([w.cols[i], [n]])
        a, b = map(int, context_bounds(seen_all[:, cols], s0, s0 + length))
        seen = seen_all[a:b][:, cols]
        enc = encode_tracks(tracks[a:b][:, cols], seen)
        ctx = decode_tracks(enc, b - a)  # exactly what the page decodes
        k = kalman(ctx, np.isfinite(ctx).all(-1), FPS)
        win = slice(s0 - a, s0 - a + length)
        sm, sv = k.smooth_x[win, :, :, 0], k.smooth_p[win, :, :, 0, 0]
        refs.append({"mean_m": np.round(sm, 9).tolist(), "var_m2": np.round(sv, 12).tolist()})
        am_np, al_np = _normalise(sm, sv)
        gaps.append(float(np.abs(am_np - w.imp_mean[i]).max() * SCALE.max()))
        # the reference uses the same rounded inputs as the page gets from clips.json
        pos = torch.as_tensor(np.round(w.pos[i : i + 1] * SCALE, 2) / SCALE, dtype=torch.float32)
        obs = torch.as_tensor(w.visible[i : i + 1])
        am = torch.as_tensor(am_np[None], dtype=torch.float32)
        al = torch.as_tensor(al_np[None], dtype=torch.float32)
        with torch.no_grad():
            mu, ls = model(torch.nan_to_num(pos) * obs[..., None], obs, types, am, al)
        clips.append(
            {
                "start_s": round(float(w.start[i] / FPS), 1),
                "camera_m": r(camera[w.start[i] : w.start[i] + w.pos.shape[1]], 2),
                "truth_m": r(w.pos[i] * SCALE, 2),
                "visible": w.visible[i].astype(int).tolist(),
                "context": {"before": s0 - a, "frames": b - a, "tracks_mm": enc},
                "python_mean": r(mu[0].numpy(), 5),
                "python_log_std": r(ls[0].numpy(), 5),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    example = tuple(
        torch.zeros(s, dtype=d)
        for s, d in (((1, 90, 23, 2), torch.float32), ((1, 90, 23), torch.float32), ((23,), torch.int64),
                     ((1, 90, 23, 2), torch.float32), ((1, 90, 23, 2), torch.float32))
    )  # fmt: skip
    torch.onnx.export(
        OnnxWrapper(model), example, str(OUT / "model.onnx"),
        input_names=["pos", "observed", "agent_type", "anchor_mean", "anchor_log_std"],
        output_names=["mean", "log_std"], opset_version=17, dynamo=False,
    )  # fmt: skip
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    audit = json.loads((ROOT / "results" / "06-baseline-audit.json").read_text())["pooled"]
    hidden = ~vis_mask & np.isfinite(tr.players).all(-1)
    present = np.isfinite(tr.players).all(-1)
    meta = {
        "match": MATCH,
        "hidden_share": round(float(hidden.sum() / present.sum()), 4),  # player positions off-camera, this match
        "hidden_gap_bins": audit["bin_fraction"],  # share of hidden positions by gap length, 8 matches (spec 06)
        "weights": args.tag,
        "fps": FPS,
        "scale_m": SCALE.tolist(),
        "kalman": {"accel_std": 3.0, "obs_std": 0.3, "min_std_m": MIN_STD_M, "clip": CLIP},
        "std_scale": args.std_scale,  # the page multiplies the model std by this factor (python_log_std excludes it)
        "agent_type": w.agent_type.tolist(),
        "attribution": "IDSSE data: Bassek et al., Scientific Data 12, 195 (2025), CC BY 4.0. "
        "Camera masks from SkillCorner open data.",
        "commit": sha.stdout.strip(),
        "clips": clips,
    }
    (OUT / "clips.json").write_text(json.dumps(meta, separators=(",", ":")))
    ref = {"note": "Python smoother (pitchvision.baselines.kalman) on the context of each clip, window frames only",
           "clips": refs}  # fmt: skip
    (ROOT / "tests" / "js" / "kalman-ref.json").write_text(json.dumps(ref, separators=(",", ":")))
    print("context-limited anchor vs the match-level anchor, max difference (m):", [round(g, 4) for g in gaps])
    print("clips at", [c["start_s"] for c in clips], "s; files:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
