"""Spec 03 erratum: score the M3 forecast of model version A on the same cells as constant velocity (spec 04 A1).

Usage: PYTHONPATH=src python scripts/rescore_03_m3.py
Needs data/models/03-masked-model.pt. Writes results/03-m3-same-cells.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from pitchvision import baselines as bl
from pitchvision.model import MaskedModel
from pitchvision.prep import FPS, metrica_game, replay_mask, skillcorner_ids, skillcorner_match
from pitchvision.windows import HALF_LENGTH, HALF_WIDTH, cut

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 40
SCALE = np.array([HALF_LENGTH, HALF_WIDTH])


@torch.no_grad()
def main() -> None:
    tr = metrica_game(3)
    vis = replay_mask(tr, skillcorner_match(skillcorner_ids()[2], ROOT / "data" / "raw" / "skillcorner"))
    w = cut(tr, vis, length=90, stride=20)
    model = MaskedModel(length=90, dim=128, heads=4, depth=4)
    model.load_state_dict(torch.load(ROOT / "data" / "models" / "03-masked-model.pt", map_location="cpu"))
    model.eval()
    obs = w.visible.copy()
    obs[:, -HORIZON:, :] = False
    types = torch.as_tensor(w.agent_type)
    mus = []
    for i in range(0, len(w.pos), 64):
        p = torch.as_tensor(w.pos[i : i + 64], dtype=torch.float32)
        mus.append(model(p, torch.as_tensor(obs[i : i + 64]), types)[0].numpy())
    mu = np.concatenate(mus)
    err_f = np.linalg.norm((mu[:, -1, :22] - w.pos[:, -1, :22]) * SCALE, axis=-1).ravel()
    cv = np.stack([bl.constant_velocity(w.pos[b, :, :22] * SCALE, obs[b, :, :22], FPS)[-1] for b in range(len(w.pos))])
    err_cv = np.linalg.norm(cv - w.pos[:, -1, :22] * SCALE, axis=-1).ravel()
    same = np.isfinite(err_cv)
    out = {
        "spec": "03-masked-model erratum (spec 04 A1)",
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "m3_cell_fraction": float(same.mean()),
        "forecast_4s_mean_m": float(err_f[same].mean()),
        "constant_velocity_4s_mean_m": float(err_cv[same].mean()),
        "forecast_4s_mean_all_cells_m": float(err_f.mean()),
        "M3_pass": bool(err_f[same].mean() < err_cv[same].mean()),
    }
    (ROOT / "results" / "03-m3-same-cells.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
