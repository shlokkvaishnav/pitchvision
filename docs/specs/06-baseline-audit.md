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

## Result (2026-11-16, Kaggle CPU, commit 57e03ad)
8 matches, 4,990,300 hidden player cells with a true position.

**Imputation (full match, pooled)**
| Baseline | Median | Mean | < 1 s | 1 to 3 s | 3 to 6 s | 6 to 9.6 s | ≥ 9.6 s (median) |
|---|---|---|---|---|---|---|---|
| linear | 3.64 m | **6.04 m** | 0.03 | 0.22 | 0.75 | 1.66 | 5.53 |
| spline | **2.48 m** | 7.84 m | 0.01 | 0.05 | 0.23 | 0.67 | 4.84 |
| kalman | 2.92 m | 8.20 m | 0.05 | 0.14 | 0.41 | 0.94 | 5.38 |

Fraction of hidden cells by gap length: 1.0%, 5.5%, 9.5%, 10.3%, **73.6%** (≥ 9.6 s).
Kalman 90% ellipse coverage: 90.3% (98.1%, 94.7%, 89.7%, 85.7% and 90.6% by bin).

- The spline has the lowest median on all 8 matches. **The imputation bar is now `spline`** (section 5).
- The spline has the highest mean of the three: it overshoots in long gaps. The median and the mean give different winners.
  Thus, the next specs report both.
- Most hidden cells (73.6%) are in gaps of 9.6 s or more. A 9 s window cannot contain these gaps.
- The Kalman 90% regions are well calibrated on the full match.

**Forecast at 4.0 s (69.8% of cells, where all four have an estimate)**
| cv_last | cv_smooth | stationary | kalman_filter |
|---|---|---|---|
| **7.14 m** | 7.45 m | 8.08 m | 7.73 m |

- The forecast bar stays `cv_last`. The smoothed velocity is worse. Thus, the velocity noise is not the cause of our high constant-velocity error.
  Our data has 69.8% usable cells and other matches than FootBots (7.27 m), so the numbers are not directly comparable.
