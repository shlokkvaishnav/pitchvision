# Spec 15, fold 2 of 5: held-out match idsse-J03WOH

Status: **complete**. One fold of five; the pooled result comes from `scripts/lomo_summary.py` once all five have run.

- Kernel: `shlokkvaishnav/pitchvision-lomo-idsse-j03woh`, status COMPLETE, run on Kaggle GPU (cuda), 6,179 s total.
- Code commit: `8f82af3` (spec 15 amendment A2), the same as fold 1. Settings: 20 epochs, batch 32, lr 3e-4, K = 20 diffusion samples, seed 0.
- Training matches: 8, tuning match `idsse-J03WOY`, test match `idsse-J03WOH`. Windows: 14,352 train, 2,803 tune, 2,703 test.
- Best tuning score: transformer at epoch 20 (the last), diffusion at epoch 19.

## Result (test match only, metres unless noted)

| Method | Median error | Mean error | Energy score | 90% coverage | Pitch-control agreement |
|---|---|---|---|---|---|
| Spline | 3.36 | 9.51 | n/a | n/a | 0.915 |
| Kalman | 3.88 | 10.02 | 8.20 | 0.885 | 0.911 |
| Centroid vote | 9.65 | 12.92 | n/a | n/a | 0.858 |
| Transformer (v1) | **2.76** | **4.91** | **3.53** | 0.911 | **0.939** |
| Diffusion (v2) | 2.98 | 5.16 | 3.74 | 0.818 | **0.939** |

Visible-only pitch-control agreement is 0.694.

Reading: the same pattern as fold 1. The transformer has lower error and a better energy score than the diffusion
model; pitch-control agreement is equal. The diffusion model's 90% regions are again too narrow (0.82 against 0.90).

## Files

- `fold-idsse-J03WOH.json`: full summary (training logs, metrics, per-gap coverage, timings).
- Not in git: the prediction arrays (`fold-idsse-J03WOH.npz`, 77 MB) and both checkpoints are in `data/15-lomo/idsse-J03WOH/` (gitignored, local only). The SkillCorner inputs are public and were not kept.
- Kernel script and metadata: `kaggle/lomo/idsse-J03WOH/`.
