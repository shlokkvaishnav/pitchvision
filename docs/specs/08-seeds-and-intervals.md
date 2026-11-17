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
