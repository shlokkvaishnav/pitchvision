# Spec 15, fold 5 of 5: held-out match metrica-2

Status: **complete**. The pooled result over the five folds is in `results/15-lomo.json` (`scripts/lomo_summary.py`).

- Kernel: `shlokkvaishnav/pitchvision-lomo-metrica-2`, status COMPLETE, run on Kaggle GPU (cuda), 5,465 s total.
- Code commit: `8f82af3` (spec 15 amendment A2), the same as folds 1 to 4. Settings: 20 epochs, batch 32, lr 3e-4, K = 20 diffusion samples, seed 0.
- Tuning match `metrica-3`, test match `metrica-2`. Windows: 17,372 train, 1,395 tune, 1,091 test.
- Best tuning score: transformer at epoch 20 (the last), diffusion at epoch 18. Diffusion sampler eta 0.

## Result (test match only, metres unless noted)

| Method | Median error | Mean error | Energy score | 90% coverage | Pitch-control agreement |
|---|---|---|---|---|---|
| Spline | **1.43** | 4.59 | n/a | n/a | 0.963 |
| Kalman | 1.79 | 5.10 | 5.50 | 0.897 | 0.958 |
| Centroid vote | 7.02 | 9.28 | n/a | n/a | 0.897 |
| Transformer (v1) | 1.45 | **2.56** | **1.86** | 0.953 | **0.971** |
| Diffusion (v2) | 1.62 | 2.81 | 2.00 | 0.876 | 0.969 |

Visible-only pitch-control agreement is 0.785.

Reading: as in fold 3, the gaps are easy and the spline has the lowest median error, just below the transformer.
The transformer has the lowest mean error, the best energy score and the highest agreement. The diffusion model is
behind the transformer on every measure; its 90% coverage is 0.88.

## Files

- `fold-metrica-2.json`: full summary (training logs, metrics, per-gap coverage, timings).
- Not in git: the prediction arrays and both checkpoints are in `data/15-lomo/metrica-2/` (gitignored, local only).
- Kernel script and metadata: `kaggle/lomo/metrica-2/`.
