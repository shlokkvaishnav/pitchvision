# Spec 07 — Final test on the locked holdout

Status: committed before any holdout match was loaded. This is the only run on the holdout.

## 1. Objective
Test the spec 05 model, unchanged, on the two IDSSE matches that the holdout hash of spec 00 locked:
`idsse-J03WN1` and `idsse-J03WOH`. These matches were never loaded by any earlier spec, script or notebook.
The rules are the final rules of the charter (section 6), made concrete below.

## 2. What is fixed
- Weights: `data/models/05-more-data.pt` (spec 05, best epoch 26). No training, no tuning.
- Std scale: s = 1.1747 (spec 05, fitted on Metrica 3).
- Windows, anchors, mask and evaluation code: as spec 05 (`scripts/train_more.py`), 10 Hz, 9 s windows, stride 2 s.
- Camera masks: the replay_ball mask. Donors: the next unused SkillCorner IDs in sorted order (9th and 10th:
  `1996435` for J03WN1, `1996436` for J03WOH).

## 3. Rules (both matches pooled)
| Rule | Pass condition |
|---|---|
| R1 Imputation | Median error on hidden player cells ≤ 0.9 × the best baseline on the same cells. Baselines (full match, one period at a time): constant velocity, linear interpolation, cubic spline, Kalman smoother, formation template. Same cells: hidden cells where every baseline has an estimate. |
| R2 Calibration | After the std scale s, the 90% ellipses contain the true position for 85% to 95% of hidden player cells. |
| R3a Forecast error | Mean error at 4.0 s is lower than the best of `cv_last`, `cv_smooth`, `stationary`, `kalman_filter` (spec 06), on the cells where all four have an estimate. |
| R3b Plausible motion | Over the 40 forecast frames, the fraction of player frames with speed > 12 m/s or acceleration > 10 m/s² is not more than in the true positions of the same cells. Speed and acceleration are finite differences at 10 Hz. |
| R4 Pipeline stability | **Not tested.** The "what if" stage was not built. |

We also report the mean imputation error of every method, and all numbers per match.

## 4. Procedure
1. A dry run of the script on development match `idsse-J03WQQ` checks the code. It does not count.
2. One run on the two holdout matches. The result goes to `results/07-holdout.json` with the git commit.
3. Whatever the result, we keep it and report it. No second run on the holdout.
