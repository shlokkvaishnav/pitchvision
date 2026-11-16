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
