# Spec 03 — Masked model, version A (Gaussian output)

Status: We committed this file before the first training run.

## 1. Objective
Train the first version of the masked model (spec 00, section 2).
This version gives a Gaussian (mean and standard deviation) for each hidden position.
It does imputation and forecast with one set of weights.
A diffusion version (spec 04) comes after this version passes.

## 2. Windows
- One window is 9.0 s at 10 Hz: 90 frames.
- A window has 23 agents: 11 home players, 11 away players and the ball.
  We use a window only if exactly 11 players of each team have a position in all 90 frames.
- Stride: 2.0 s between windows.
- Coordinates: x / 52.5 and y / 34. Thus, the pitch is from −1 to 1 on both axes.

## 3. Tasks in training
For each window, we select one task with probability 0.5:
- **Imputation.** The input is the camera-visible cells in all 90 frames. The mask is the footprint replay of spec 01, amendment A1. The ball is always visible.
- **Forecast.** The same, but the last 40 frames (4.0 s) are hidden for all agents.

The loss is the Gaussian negative log-likelihood on all hidden cells that have a true position.
Augmentation: random flip of x. A flip of x does not change the camera side. We do not flip y, because the camera is on one side.

## 4. Model
- Input per cell: observed position (zero if hidden), an observed flag, an agent-type embedding (home, away, ball), a time embedding.
- Body: 4 blocks. Each block has attention over time (for each agent) and attention over agents (for each frame).
  Width 128, 4 heads. There is no agent-slot embedding. Thus, the model does not depend on the order of the players.
- Output per cell: mean (x, y) and log standard deviation (x, y).

## 5. Data split
Development data only (spec 00, section 4). Metrica games 1 and 2 for training, game 3 for validation.
When we have the PFF files, an amendment adds them.

## 6. Rules (validation game 3)
| Rule | Pass condition |
|---|---|
| M1 Imputation | The median error on hidden player cells is at most 0.9 × the error of the bar baseline from spec 02 on the same cells. The baseline uses the full match. Thus, this comparison favours the baseline. |
| M2 Calibration | The 90% ellipse (chi-square, 2 degrees of freedom) contains the true position for 85% to 95% of hidden player cells. |
| M3 Forecast | The mean error at 4.0 s is lower than constant velocity on the same cells. |

These rules use development data. They are not the final rules R1 to R3, which use the holdout.

## 7. Compute
Unit tests run on CPU with a small model. Training runs on a Kaggle GPU.

## 8. Output
`results/03-masked-model.json`: the metrics, the rule results, the training settings and the git commit SHA.

## Result (2026-11-16, Kaggle GPU, commit 426c57c)
Training: 2,307 windows (Metrica games 1 and 2), 30 epochs, 843 s. The loss went down in all epochs.
Validation: game 3, 1,395 windows, 1,318,094 hidden player cells.

| Rule | Model | Bar | Result |
|---|---|---|---|
| M1 Imputation (median error) | 5.39 m | Kalman smoother 2.13 m on the same cells (pass needs ≤ 1.91 m) | Fail |
| M2 Calibration (90% ellipses) | 88.4% | 85% to 95% | Pass |
| M3 Forecast at 4.0 s (mean error) | 12.4 m | Constant velocity 8.6 m | Fail |

**Interpretation (hypotheses for version B, not tested).**
1. Context. The model sees 9 s. The Kalman smoother sees the full match. With the replay mask, 49% of cells are hidden,
   and the median hidden run is 4.7 s. Thus, a player is often hidden for most of a window, and the model has little information about that player.
2. Data. 2,307 training windows is small for a transformer with 4 blocks. The PFF data (64 matches) is not available yet.
3. The model knows its uncertainty well (M2 passes), but its mean is not good enough.

A version B needs a new spec, written before its run. The model weights are in `data/models/` (not in git).

## Erratum (2026-11-16)
The M3 comparison above is not on the same cells. The constant-velocity mean removed the cells where the baseline has no estimate
(players hidden in all observed frames of the window), but the model mean kept them. See spec 04, amendment A1.

**Corrected M3** (`scripts/rescore_03_m3.py`, `results/03-m3-same-cells.json`, the same weights). 73.9% of the cells at 4.0 s have a baseline estimate.
| M3 Forecast at 4.0 s (mean error, same cells) | Model 7.42 m | Constant velocity 8.62 m | **Pass** |
|---|---|---|---|

On all cells, the model error is 12.4 m (the value above). Thus, version A passes M2 and M3, and it fails M1.
