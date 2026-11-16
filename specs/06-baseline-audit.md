# Spec 06 — Baseline audit on all development matches

Status: We committed this file before the first run.

## 1. Objective
The deep research of 2026-11-16 found three risks in our baselines:
1. In MIDAS (arXiv 2408.10878), on Metrica with a 50% camera mask, a cubic spline (mean error 1.92) is much better than linear
   interpolation (3.11). We did not test a spline. Thus, the Kalman smoother is maybe not the best classical bar.
2. Our constant-velocity forecast error at 4.0 s (8.62 m) is higher than in FootBots (7.27 m, other data).
   Our velocity uses only the last two visible frames. This is noisy.
3. In Choi (arXiv 2607.11548), 50% to 57% of hidden player observations are in gaps longer than 9.6 s. Our windows are 9 s.

This spec measures the classical baselines on all 8 development matches. It trains no model. It has no pass/fail rule.
It fixes the bars for the next specs (section 5).

## 2. Matches and masks
Metrica 1, 2, 3 and IDSSE J03WMX, J03WOY, J03WPY, J03WQQ, J03WR9 (spec 05), 10 Hz, with the replay_ball mask and the donors of spec 05.

## 3. Imputation baselines (full match, one period at a time)
| Name | Content |
|---|---|
| `linear` | Linear interpolation between visible frames (spec 02). |
| `spline` | Cubic spline through all visible frames of the player in the period (`scipy.interpolate.CubicSpline`, natural boundary). Outside the first and last visible frame: hold. |
| `kalman` | RTS smoother of spec 02. Also its 90% ellipse coverage (from the smoother covariance). |

Metrics on the hidden player cells with a true position: median and mean error (m), for all cells and for 5 bins of the
length of the hidden run: < 1 s, 1 to 3 s, 3 to 6 s, 6 to 9.6 s, ≥ 9.6 s. Also the fraction of hidden cells in each bin.

## 4. Forecast baselines (windows of spec 04, last 40 frames hidden, error at 4.0 s)
| Name | Content |
|---|---|
| `cv_last` | Constant velocity of spec 02 (velocity from the last two visible frames, speed cap 8 m/s). |
| `cv_smooth` | The same, but the velocity is the mean over the last 5 consecutive visible frames (4 steps). If fewer than 5, the last two frames. |
| `stationary` | Hold the last visible position. |
| `kalman_filter` | The forecast anchor of spec 04 (causal filter, prediction, speed cap, pitch clip). |

Metric: mean error at 4.0 s on the cells where all four baselines have an estimate.

## 5. Decisions (fixed before the run)
- The imputation bar for the next specs is the baseline with the lowest pooled median error.
- The forecast bar for the next specs is the baseline with the lowest pooled mean error at 4.0 s.
- If a bar changes, the spec 03 to 05 results stay as they are. We add a note to them.

## 6. Compute and output
CPU only (Kaggle CPU kernel, because the IDSSE files are large). `results/06-baseline-audit.json`: the metrics per match and pooled, and the git commit SHA.
