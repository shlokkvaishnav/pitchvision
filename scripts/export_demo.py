"""Export the in-browser demo: the anchored model as ONNX and three clips of IDSSE match J03WQQ.

Usage: PYTHONPATH=src python scripts/export_demo.py --weights data/models/04-anchored-model.pt --tag spec04
Writes site/demo/model.onnx and site/demo/clips.json.

Clips: the windows at 1/4, 1/2 and 3/4 of the match (a fixed rule, no selection by result).
J03WQQ is a test match of spec 05, not a holdout match. IDSSE data: Bassek et al. (2025), CC BY 4.0.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_more as tm

from pitchvision.anchor import SCALE
from pitchvision.model import AnchoredModel
from pitchvision.prep import replay_corners, skillcorner_match

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "demo"
MATCH = "idsse-J03WQQ"


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
    args = ap.parse_args()

    model = AnchoredModel(length=90, dim=128, heads=4, depth=4)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    tr, vis_mask, w = tm.load_match(MATCH, tm.donors()[MATCH], Path(args.raw))
    camera = replay_corners(tr, skillcorner_match(tm.donors()[MATCH], Path(args.raw)))  # (T, 4, 2) metres
    picks = [len(w.pos) // 4, len(w.pos) // 2, 3 * len(w.pos) // 4]
    types = torch.as_tensor(w.agent_type)
    clips = []
    for i in picks:
        pos = torch.as_tensor(w.pos[i : i + 1], dtype=torch.float32)
        obs = torch.as_tensor(w.visible[i : i + 1])
        am = torch.as_tensor(w.imp_mean[i : i + 1], dtype=torch.float32)
        al = torch.as_tensor(w.imp_ls[i : i + 1], dtype=torch.float32)
        with torch.no_grad():
            mu, ls = model(torch.nan_to_num(pos) * obs[..., None], obs, types, am, al)
        clips.append(
            {
                "start_s": round(float(w.start[i] / tm.FPS), 1),
                "camera_m": r(camera[w.start[i] : w.start[i] + w.pos.shape[1]], 2),
                "truth_m": r(w.pos[i] * SCALE, 2),
                "visible": w.visible[i].astype(int).tolist(),
                "anchor_mean": r(w.imp_mean[i], 5),
                "anchor_log_std": r(w.imp_ls[i], 5),
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
        "fps": tm.FPS,
        "scale_m": SCALE.tolist(),
        "agent_type": w.agent_type.tolist(),
        "attribution": "IDSSE data: Bassek et al., Scientific Data 12, 195 (2025), CC BY 4.0. "
        "Camera masks from SkillCorner open data.",
        "commit": sha.stdout.strip(),
        "clips": clips,
    }
    (OUT / "clips.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("clips at", [c["start_s"] for c in clips], "s; files:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
