# Spec 15, fold 3 of 5: held-out match idsse-J03WPY

Status: **complete**. One fold of five; the pooled result comes from `scripts/lomo_summary.py` once all five have run.

- Kernel: `shlokkvaishnav/pitchvision-lomo-idsse-j03wpy`, status COMPLETE, run on Kaggle GPU (cuda), 6,460 s total.
- Code commit: `8f82af3` (spec 15 amendment A2), the same as folds 1 and 2. Settings: 20 epochs, batch 32, lr 3e-4, K = 20 diffusion samples, seed 0.
- Tuning match `idsse-J03WQQ`, test match `idsse-J03WPY`. Windows: 15,200 train, 1,778 tune, 2,880 test.
- Best tuning score: transformer at epoch 16, diffusion at epoch 19. Diffusion sampler eta 0.

## Result (test match only, metres unless noted)

| Method | Median error | Mean error | Energy score | 90% coverage | Pitch-control agreement |
|---|---|---|---|---|---|
| Spline | **1.81** | 5.79 | n/a | n/a | 0.947 |
| Kalman | 2.18 | 6.21 | 4.94 | 0.903 | 0.945 |
| Centroid vote | 7.43 | 10.22 | n/a | n/a | 0.877 |
| Transformer (v1) | 1.84 | **3.54** | **2.57** | 0.917 | **0.960** |
| Diffusion (v2) | 2.05 | 3.75 | 2.71 | 0.863 | 0.958 |

Visible-only pitch-control agreement is 0.745.

Reading: on this match the gaps are easier (every method has lower error than in folds 1 and 2). The spline has the
lowest median error, just below the transformer; the transformer has the lowest mean error, the best energy score and
the highest pitch-control agreement. The diffusion model is behind the transformer on every measure, and its 90%
regions are again too narrow (0.86 against 0.90).

## Files

- `fold-idsse-J03WPY.json`: full summary (training logs, metrics, per-gap coverage, timings).
- Not in git: the prediction arrays and both checkpoints are in `data/15-lomo/idsse-J03WPY/` (gitignored, local only). The SkillCorner inputs are public and were not kept.
- Kernel script and metadata: `kaggle/lomo/idsse-J03WPY/`.
