# Spec 05 — Masked model B with more data, early stopping and calibration

Status: We committed this file before the first training run.

## 1. Objective
Version B (spec 04) passed the forecast rule (M3). It failed imputation (M1: 2.24 m against 2.13 m) and calibration
(M2: 84.1%). Before training, version B is equal to the Kalman anchor. After training, it was worse than the anchor.
Our hypothesis: overfitting on 2 training matches (2,307 windows).

This spec tests three changes together:
1. **More data.** The IDSSE dataset (Bassek et al., Scientific Data 12, 195, 2025; CC BY 4.0): 7 full Bundesliga matches
   with positions of all players and the ball, 25 Hz. kloppy loads it (`sportec.load_open_tracking_data`).
2. **Early stopping.** We evaluate after each epoch on a tuning match, and we keep the best epoch. Epoch 0 (the anchor) is a candidate.
3. **Calibration.** One scale factor s for the standard deviation, fitted on the tuning match (closed form: s² = mean of z² over the hidden cells and both axes).

We do not change the model, the windows, the tasks or the training settings of spec 04.

## 2. Matches
Match IDs: `metrica-<game>` and `idsse-<ID>`. The holdout hash of spec 00 puts `idsse-J03WN1` and `idsse-J03WOH` in the holdout.
We do not load them. The 5 development matches, sorted: J03WMX, J03WOY, J03WPY, J03WQQ, J03WR9.

| Role | Matches |
|---|---|
| Train | Metrica 1, Metrica 2, IDSSE J03WMX, J03WOY, J03WPY |
| Tuning (early stopping and calibration) | Metrica 3 (it was the validation match of specs 03 and 04, thus it is not a fresh test) |
| Test (rules) | IDSSE J03WQQ, J03WR9 (no model saw them) |

Each IDSSE match is resampled to 10 Hz. The camera mask: the replay_ball mask of spec 02 A1.
Donors: the sorted SkillCorner IDs. Metrica 1 to 3 keep donors 1 to 3. The IDSSE matches get donors 4 to 8, in the sorted order above.

## 3. Early stopping
After each epoch (and before the first epoch), we compute the median imputation error on the hidden player cells of the tuning match.
We keep the weights of the epoch with the lowest error. Maximum: 30 epochs.

## 4. Rules (test matches J03WQQ and J03WR9, pooled)
| Rule | Pass condition |
|---|---|
| M1 Imputation | The median error on hidden player cells is at most 0.9 × the error of the Kalman smoother (full match) on the same cells. |
| M2 Calibration | After the scale factor, the 90% ellipses contain the true position for 85% to 95% of hidden player cells. |
| M3 Forecast | The mean error at 4.0 s is lower than constant velocity, on the cells where constant velocity has an estimate (spec 04 A1). |

We also report: the selected epoch, the scale factor, the anchor alone, and all metrics on the tuning match.

## 5. Compute and output
Training on a Kaggle GPU. `results/05-more-data.json`: the metrics, the rule results, the settings and the git commit SHA.

## Note (2026-11-16, after spec 06)
Spec 06 found a stronger classical bar for imputation: a cubic spline (pooled median 2.48 m against 2.92 m for the
Kalman smoother on the full matches). The rules of this spec stay as written. But a model that passes M1 here must
also beat the spline before we say that it beats the classical baselines.

## Result (2026-11-17, Kaggle GPU, commit 22e0208)
Training: 11,873 windows (5 matches), 30 epochs, 3,997 s. Best epoch on the tuning match: 26. Scale factor s = 1.175.
Test: IDSSE J03WQQ and J03WR9, 4,679 windows, 4,554,291 hidden player cells.

| Rule | Model | Anchor alone | Bar | Result |
|---|---|---|---|---|
| M1 Imputation (median error) | **2.43 m** | 3.25 m | ≤ 2.95 m (0.9 × Kalman 3.27 m) | **Pass** (ratio 0.74) |
| M2 Calibration (90% ellipses, after s) | **92.1%** | 88.8% | 85% to 95% | **Pass** |
| M3 Forecast at 4.0 s (mean error, same cells, 68.3%) | **5.17 m** | 7.44 m | below 6.94 m (constant velocity) | **Pass** |

Tuning match (Metrica 3): imputation 1.94 m (Kalman 2.13 m), coverage 88.9%, forecast 6.29 m (constant velocity 8.62 m).
On all test cells, the forecast error at 4.0 s is 7.38 m.

**Spline bar (note of 2026-11-16).** `scripts/rescore_05_spline.py` scores the model, the Kalman smoother and the cubic spline
of spec 06 on the same test cells (CPU; the model median is the same as on the GPU, 2.433 m). `results/05-spline-same-cells.json`:

| Same test cells | Median | Mean |
|---|---|---|
| Model | **2.43 m** | **4.58 m** |
| Cubic spline | 2.82 m | 11.24 m |
| Kalman smoother | 3.27 m | 9.66 m |

The model is 14% lower than the spline on the median and 59% lower on the mean. Thus, it also beats the strongest classical bar.

**Interpretation.**
- **The hypothesis of spec 04 is supported.** With 5 training matches instead of 2, the tuning error goes down after epoch 12,
  and it goes below the anchor (epoch 0). In spec 04, training made the model worse than the anchor.
- **Long gaps.** The large drop of the mean (11.24 m to 4.58 m) shows that the model removes most of the large errors of the
  spline and the Kalman smoother in long gaps.
- **Limits.** Two test matches from one league. The windows overlap (stride 2 s), thus one cell can count more than once
  (the same for all methods). The holdout matches (J03WN1, J03WOH) are not used yet.

**Consequence.** The imputation passes on new matches. The demo on the site now uses these weights, with the scale factor.
