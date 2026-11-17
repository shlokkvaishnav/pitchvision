# Spec 08 — Seed variation and confidence intervals

Status: committed before the runs. Report only: no rule, no new decision, and the model on the site does not change.

## 1. Objective
The spec 05 and spec 07 results come from one training seed, and they have no intervals.
This spec measures how much the results move (A) with the training seed and (B) with the sample of match minutes.

## 2. Part A: seeds
- Train the spec 05 setup again with seeds 1 and 2: the same matches, settings, early stopping and calibration
  (`scripts/train_more.py --seed 1|2`, Kaggle GPU).
- Evaluate each seed on the spec 05 test matches (J03WQQ, J03WR9), as in spec 05. The holdout is not used.
- Report each metric for seeds 0, 1 and 2, with the mean and the range.

## 3. Part B: intervals
- For the seed 0 model (the model on the site), on (i) the spec 05 test matches and (ii) the spec 07 locked matches.
  For (ii), this re-scores the same frozen model. It adds intervals only and changes no result or decision.
- Block bootstrap: each match is cut into 5-minute blocks (a window belongs to the block of its first frame).
  2,000 resamples of blocks with replacement, seed 0. 95% percentile intervals.
- Quantities: median imputation error of the model and of the cubic spline, and their ratio; 90% coverage (after the
  spec 05 scale factor); mean 4 s forecast error of the model and of `cv_last`, and their difference
  (on the cells where `cv_last` has an estimate).
- Medians are computed from 1 cm histograms per block.

## 4. Output
`results/08-seeds.json` (part A) and `results/08-intervals.json` (part B), with the git commit.

## Result, part B (2026-11-17)
95% block-bootstrap intervals (5-minute blocks, 2,000 resamples). The point estimates equal the spec 05 and spec 07
results (medians to the 1 cm histogram bin).

| | Spec 05 test (2 matches, 32 blocks) | Spec 07 locked (2 matches, 21 blocks) |
|---|---|---|
| Model median error | 2.44 m [2.18, 2.78] | 2.71 m [2.40, 3.07] |
| Spline median error | 2.82 m [2.37, 3.43] | 3.28 m [2.80, 3.90] |
| Ratio model / spline | 0.87 [0.80, 0.93] | 0.83 [0.78, 0.87] |
| 90% coverage | 92.1% [91.4, 92.7] | 92.4% [91.5, 93.2] |
| Forecast at 4 s, model | 5.17 m [4.97, 5.38] | 5.18 m [4.89, 5.47] |
| Forecast at 4 s, cv_last | 6.94 m [6.56, 7.29] | 6.97 m [6.47, 7.48] |
| Difference (model − cv_last) | −1.76 m [−1.94, −1.59] | −1.79 m [−2.02, −1.56] |

**Interpretation.** On both test sets, the whole interval of the error ratio is below 1, and the whole interval of the
forecast difference is below 0. Coverage stays inside 85 to 95%. The intervals describe variation over match minutes
within these 4 matches. They do not cover variation between leagues or seasons.

## Result, part A (2026-11-17)
The spec 05 setup trained 3 times (seeds 0, 1, 2), each evaluated on the spec 05 test matches (J03WQQ, J03WR9).
Seed 0 is the model on the site. Seeds 1 and 2 ran on Kaggle (one kernel, commit 75b90cc); raw outputs are in
`results/08-seeds/`.

| | Seed 0 | Seed 1 | Seed 2 | Mean | Baseline |
|---|---|---|---|---|---|
| Imputation median error | 2.43 m | 2.40 m | 2.37 m | 2.40 m | Kalman 3.27 m |
| Ratio to Kalman | 0.74 | 0.73 | 0.72 | 0.73 | — |
| 90% coverage | 92.1% | 91.9% | 91.7% | 91.9% | — |
| Forecast at 4 s, mean error | 5.17 m | 5.17 m | 5.14 m | 5.16 m | constant velocity 6.94 m |
| Best epoch | 26 | 30 | 29 | | |
| Std scale factor | 1.175 | 1.181 | 1.161 | 1.172 | |

**Interpretation.** All 3 seeds pass all spec 05 rules. The spread over seeds (0.06 m in median imputation error, 0.03 m
in forecast error, 0.4 points in coverage) is much smaller than the gap to the baselines and smaller than the part B
intervals. The site model (seed 0) is the weakest of the 3 on imputation, so the headline numbers are not a lucky seed.
Seeds 1 and 2 had their best tuning score in the last 2 epochs, so 30 epochs may be slightly short; this was not changed.
