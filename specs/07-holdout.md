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

## Amendment A1 (2026-11-17, after the dry run, before any holdout match was loaded)
The dry run on development match J03WQQ found that the forecast mean wobbles by about 14 cm from frame to frame.
Speeds are fine, but at 10 Hz this gives accelerations above 10 m/s² in about 30% of forecast frames (true positions: 0.03%).
The rules do not change. We add one **report-only** number: R3a and R3b again after a fixed 9-frame moving average
over time on the forecast (no learned parameters). On J03WQQ (full dry run) it lowered the flagged fraction from 34.3% to 1.6% and the 4 s error from 5.40 m to 5.39 m.
We chose the 9 frames on J03WQQ only.

## Result (2026-11-17, CPU, commit d6b2f1a) — the one holdout run
2 matches, 2,911 windows, 2,931,610 hidden player cells. All baselines have an estimate on every cell.

| Rule | Model | Best baseline | Bar | Result |
|---|---|---|---|---|
| R1 Imputation (median error) | **2.70 m** | spline 3.28 m | ≤ 2.95 m (0.9 × spline) | **Pass** (ratio 0.82) |
| R2 Calibration (90% ellipses) | **92.4%** | — | 85% to 95% | **Pass** |
| R3a Forecast at 4.0 s (mean error, 66.5% of cells) | **5.18 m** | cv_last 6.97 m | lower | **Pass** |
| R3b Plausible motion (forecast frames flagged) | 38.2% | true positions 0.07% | not more than truth | **Fail** |
| R4 Pipeline stability | — | — | — | Not tested |

Mean imputation error: model 4.78 m, linear 6.93 m, spline 9.32 m, Kalman 9.86 m.
Report only (A1): with the 9-frame moving average, the flagged fraction is 2.3% and the 4 s error is 5.18 m.

| Per match | Windows | Model median | Spline median | Coverage | Forecast (model / cv_last) |
|---|---|---|---|---|---|
| J03WN1 | 208 | 2.27 m | 2.30 m | 92.6% | 4.73 / 6.09 m |
| J03WOH | 2,703 | 2.73 m | 3.36 m | 92.3% | 5.22 / 7.05 m |

**Interpretation.**
- **Imputation and calibration hold on unseen matches.** The margin over the spline (18%) is larger than on the spec 05 test matches (14%).
- **The pooled result is mostly one match.** In J03WN1 the away team has 10 players for about 92% of the match (a red card).
  The windows need 11 players per team, so only 208 windows qualify. On that match the model is only 1% better than the spline.
  The model cannot handle a team with fewer than 11 players.
- **The forecast is accurate but not smooth.** The mean wobbles from frame to frame, so the raw forecast fails the motion rule.
  A fixed moving average removes most of it without a loss of accuracy, but it is still above the rate of real players.
- **Limits.** 2 holdout matches, one league, one training seed. The holdout is now used. No more tests on it.
