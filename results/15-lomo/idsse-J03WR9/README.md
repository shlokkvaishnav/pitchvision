# Spec 15, fold 4 of 5: held-out match idsse-J03WR9

Status: **complete**. One fold of five; the pooled result comes from `scripts/lomo_summary.py` once all five have run.

- Kernel: `shlokkvaishnav/pitchvision-lomo-idsse-j03wr9`, status COMPLETE, run on Kaggle GPU (cuda), 6,425 s total.
- Code commit: `8f82af3` (spec 15 amendment A2), the same as folds 1 to 3. Settings: 20 epochs, batch 32, lr 3e-4, K = 20 diffusion samples, seed 0.
- Tuning match `metrica-1`, test match `idsse-J03WR9`. Windows: 15,741 train, 1,216 tune, 2,901 test.
- Best tuning score: transformer at epoch 18, diffusion at epoch 19. Diffusion sampler eta 0.

## Result (test match only, metres unless noted)

| Method | Median error | Mean error | Energy score | 90% coverage | Pitch-control agreement |
|---|---|---|---|---|---|
| Spline | 3.42 | 14.25 | n/a | n/a | 0.913 |
| Kalman | 3.91 | 11.50 | 10.68 | 0.899 | 0.902 |
| Centroid vote | 9.71 | 13.18 | n/a | n/a | 0.853 |
| Transformer (v1) | **2.77** | **5.16** | **3.73** | 0.873 | **0.935** |
| Diffusion (v2) | 3.01 | 5.51 | 4.03 | 0.815 | 0.922 |

Visible-only pitch-control agreement is 0.720.

Reading: the transformer is best on every error and agreement measure. The diffusion model is second and ahead of the
classical methods. Both learned models' 90% regions are too narrow on this match (transformer 0.87, diffusion 0.82).

## Files

- `fold-idsse-J03WR9.json`: full summary (training logs, metrics, per-gap coverage, timings).
- Not in git: the prediction arrays and both checkpoints are in `data/15-lomo/idsse-J03WR9/` (gitignored, local only). The SkillCorner inputs are public and were not kept.
- Kernel script and metadata: `kaggle/lomo/idsse-J03WR9/`.
