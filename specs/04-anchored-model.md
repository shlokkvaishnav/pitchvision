# Spec 04 — Masked model, version B (Kalman anchor)

Status: We committed this file before the first training run.

## 1. Objective
Version A (spec 03) was worse than the Kalman smoother (5.39 m against 2.13 m). The 90% regions were correct (88.4%).
Our hypothesis: a window of 9 s does not have sufficient information, because the median hidden run is 4.7 s.
The Kalman smoother uses the full match. Version B gives the model the Kalman estimate as an input,
and the model learns only a correction to it. When the correction is zero, the model is equal to the Kalman estimate.

Only the input and the output head change. The windows, the tasks, the body of the model, the training settings
and the data split of spec 03 stay the same.

## 2. Anchor
We run the Kalman filter of spec 02 (constant velocity, `accel_std` = 3, `obs_std` = 0.3) on the full match.
The input is the camera-visible cells only. The ball is a 23rd track, and it is always visible.
| Task | Anchor mean and standard deviation |
|---|---|
| Imputation | The RTS smoother (all visible cells of the match, before and after the window). |
| Forecast | Causal: in the first 50 frames, the filter (cells up to that frame). In the last 40 frames, the prediction from the filter state at frame 50, with the speed clipped to 8 m/s. Thus, the anchor does not use the future. |

The standard deviation is the square root of the position variance of the filter or smoother, on each axis.

## 3. Model
- Input per cell: as spec 03, plus the anchor position and the log of the anchor standard deviation (both normalised as the positions).
- Output: mean = anchor + correction, and log standard deviation = log anchor standard deviation + correction.
- The output layer starts at zero. Thus, before training, the model gives the anchor.

## 4. Rules (validation game 3)
The rules M1, M2 and M3 of spec 03, with the same bars. We also report the anchor alone (report only).

## 5. Output
`results/04-anchored-model.json`: the metrics, the anchor metrics, the rule results, the training settings and the git commit SHA.

## Amendment A1 (2026-11-16, before the training run)
A CPU smoke test (1 epoch, 16 windows) showed two problems. We found them before the training run.
1. **M3 cells.** The constant-velocity baseline has no estimate for a player that is hidden in all observed frames of the window
   (26% of the cells at 4.0 s). The mean of spec 03 removed these cells for the baseline only. Thus, the two errors were not on the same cells.
   From now on, M3 uses only the cells where the baseline has an estimate, for the model and for the baseline. We also report the error of the model on all cells.
2. **Anchor drift.** For a player that is hidden for a long time, the Kalman prediction moves far from the pitch (mean error 45 m at 4.0 s on these cells).
   We clip the anchor mean to the pitch plus a margin of 5% (−1.05 to 1.05 after normalisation).

## Result (2026-11-16, Kaggle GPU, commit 0e8ad5e)
Training: 2,307 windows, 30 epochs, 817 s. The loss went down from −1.71 to −2.34.
Validation: game 3, 1,395 windows, 1,318,094 hidden player cells.

| Rule | Model B | Anchor alone | Bar | Result |
|---|---|---|---|---|
| M1 Imputation (median error) | 2.24 m | 2.13 m | ≤ 1.91 m (0.9 × Kalman 2.13 m) | Fail |
| M2 Calibration (90% ellipses) | 84.1% | 87.2% | 85% to 95% | Fail |
| M3 Forecast at 4.0 s (mean error, same cells, 73.9%) | **6.64 m** | 9.38 m | below 8.62 m (constant velocity) | **Pass** |

On all cells, the forecast error at 4.0 s is 8.93 m (spec 03: 12.4 m).

**Interpretation.**
- **Forecast.** The correction is large and useful: 6.64 m against 9.38 m for the anchor, and 23% lower than constant velocity.
  Version A got 7.42 m on the same cells. Thus, the anchor helps the forecast.
- **Imputation.** The correction makes the anchor a little worse (2.24 m against 2.13 m), and the ellipses become too small (84.1%).
  The training loss goes down in all epochs, but the validation error goes up. This is overfitting on 2 matches.
  In imputation, the Kalman smoother already uses the full match, so the model has little to add with 2,307 windows.

**Consequence.** M1 fails. The "what if" stage stays blocked. The next lever is more data (PFF FC World Cup 2022, 64 matches).
We do not tune version B on game 3, because each new try on the same validation game increases the risk of a fit to it.
