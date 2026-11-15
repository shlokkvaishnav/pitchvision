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
