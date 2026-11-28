# Spec 15 — Leave-one-match-out cross-validation with a diffusion model

Status: registered on 2026-11-24, before any fold was run and before the diffusion model was trained.
Method choice: [spec 14](14-literature-check.md). Decisions approved by the owner on 2026-11-24.

**Amended by A2 (2026-11-24, before any fold ran): 5 folds, not 10.** The headline is 5-fold leave-one-match-out
cross-validation over a 10-match pool: 5 matches, chosen by a fixed rule, are each held out once; every fold trains on
the other 9. Where this spec says 10 folds or 10 per-match values, read 5 (see A2 at the end).

**This is cross-validation, not a fresh locked test.** All 10 matches have been looked at by earlier specs, including
the two spec 07 holdout matches (J03WN1, J03WOH), which are used here on the owner's decision. The numbers below say
how well each method does on a match it was not trained on, averaged over 10 such matches. They are not a final test
on untouched data, and the docs must say so wherever they are quoted.

## 1. Question
Does a diffusion model that samples K plausible paths per hidden player place hidden players better than the classical
methods, the published training-free method (Choi 2026), and our axial transformer, and are its 90% regions calibrated
at every gap length? And does it give more of the pitch-control picture back?

## 2. Data and folds
- Matches (sorted): `idsse-J03WMX`, `idsse-J03WN1`, `idsse-J03WOH`, `idsse-J03WOY`, `idsse-J03WPY`, `idsse-J03WQQ`,
  `idsse-J03WR9`, `metrica-1`, `metrica-2`, `metrica-3`. No new data.
- Camera masks: the replay_ball mask with the same SkillCorner donors as before (`pitchvision.dataset.donors` for the
  8 development matches; `1996435` for J03WN1 and `1996436` for J03WOH, as spec 07).
- Windows, anchors: as spec 05 (`load_match`: 9 s windows at 10 Hz, stride 2 s, 11 v 11 only).
- **10 folds.** Fold *i* tests on match *i* of the list. Its **tuning match** is the next match after *i* in the list
  (cyclic), skipping J03WN1 (only 208 windows, too few to tune on). The other 8 matches are the training matches.
  The tuning match is used for early stopping, the transformer's std scale and the diffusion sampler's eta, nothing else.
  The test match is not loaded until its model is final.

## 3. Methods (every fold)
| Name | What | Distribution |
|---|---|---|
| `spline` | Natural cubic spline per period (spec 06) | point |
| `kalman` | Constant-velocity Kalman smoother per period (spec 02), mean and variance | Gaussian |
| `centroid_vote` | Role-anchored centroid voting (Choi 2026, B4), online, per period, 10 Hz | point |
| `transformer` | Spec 05 model B retrained: AnchoredModel(dim 128, heads 4, depth 4), AdamW lr 3e-4, batch 32, one-cycle, imputation and forecast 50/50, x-flip; 20 epochs, keep the epoch with the lowest tuning-match median error; std scale fitted on the tuning match (closed form) | Gaussian |
| `diffusion` | New, section 4 | K = 20 samples |

`centroid_vote` follows `src/impute_bench.py` of Choi's code: per team, each visible player with a stored role offset
votes for the team centre as (position − offset); with at least 3 votes the centre is the mean vote, else the visible
mean. Offsets are an EMA of (position − centre) for visible players. Hidden player = centre + offset; if the player has
no EMA offset yet, the last visible-centre offset; if none, the last seen position. Choi's EMA weight is 0.1 per 0.2 s
step; at 10 Hz we use 1 − 0.9^0.5 per frame (same time constant). No last-seen decay toward the team mean (B1) beyond this.

20 epochs instead of spec 05's 30: with 8 training matches instead of 5, 20 epochs is about the same number of steps,
and it keeps the 10 folds inside the GPU budget. Both trained models use fp16 autocast on the GPU.

## 4. Diffusion model
CSDI / U2Diff recipe on the axial backbone (spec 14, section 3).
- **Target:** the residual of hidden player positions from the Kalman anchor (`imp_mean`), in units of 5 m.
- **Input per cell:** the noisy residual (0 on observed cells), the observed position (0 if hidden), the observed flag,
  the anchor mean and log-std (`imp_mean`, `imp_ls`); plus agent type, time and a diffusion-step embedding.
- **Denoiser:** 4 axial blocks (time attention, then agent attention, then an MLP), dim 128, 4 heads, the same block as
  the transformer. Output: the predicted noise (2 values per cell).
- **Noise schedule:** 50 steps, β quadratic from 1e-4 to 0.5 (CSDI, U2Diff).
- **Training:** noise-prediction MSE on hidden player cells, real camera masks, x-flip; AdamW lr 3e-4, batch 32, one-cycle,
  20 epochs. Early stopping: keep the epoch with the lowest tuning-match denoising loss (fixed noise and steps, seed 0).
- **Sampling:** DDIM with 10 steps, K = 20 samples per window. eta is chosen from {0, 0.5, 1} on the tuning match by the
  lowest mean energy score over 256 evenly spaced tuning windows.
- **Point estimate:** the mean of the 20 samples (primary). The coordinate-wise median is reported too.
- **90% region:** the Gaussian prediction ellipse from the 20 samples (sample mean m, sample covariance S): the truth y is
  inside if (K/(K+1)) (y − m)ᵀ S⁻¹ (y − m) ≤ 2(K−1)/(K−2) · F₀.₉(2, K−2). This is the exact 90% region for a new draw when
  the samples are Gaussian; no scale is fitted (no conformal step).

## 5. What is scored
- **Cells:** hidden player cells of every test-match window (counted once per window, as all earlier specs), where every
  method has an estimate.
- **Gap length:** the length of the hidden run (per period, full match) that contains the cell. Buckets: < 1 s, 1–3 s,
  3–6 s, 6–9.6 s, ≥ 9.6 s.
- **Point error:** median and mean distance (m) to the truth.
- **Energy score** (2-D CRPS, lower is better): ES = mean‖Xₖ − y‖ − ½ mean over k ≠ j of ‖Xₖ − Xⱼ‖. Gaussian methods use
  20 draws (seed 0); point methods have ES = error.
- **Coverage:** share of cells inside the 90% region (kalman, transformer, diffusion), overall and per gap bucket.
- **Sample spread check (diffusion):** mean distance of samples to their mean per gap bucket (must be > 0 and should rise
  with the gap); Spearman correlation between spread and error per match.
- **Pitch control** (spec 13 setup): the middle frame (45 of 90) of every test window with a hidden player; share of the
  32-column grid where the controlling team (p ≥ 0.5) is the same as with the truth. Versions: `visible` (hidden players
  left out), `spline`, `kalman`, `centroid_vote`, `transformer` (mean), `diffusion` (sample mean). Velocities from each
  version's own positions 0.5 s earlier. Recovered share = (version − visible) / (truth − visible).

## 6. Reporting
- Every number per test match, and pooled two ways: **match-weighted** (the mean of the 10 per-match values) and
  **position-weighted** (all cells of all folds together).
- 95% intervals: block bootstrap over 5-minute stretches of play, 2,000 samples, seed 0; blocks are resampled within each
  match (stratified), so every bootstrap sample has all 10 matches. Medians from 1 cm histograms (as spec 08).
- "Best classical method" = the lowest position-weighted pooled median among `spline`, `kalman`, `centroid_vote`.

## 7. Claims and the rules for making them
| ID | Claim | Made only if |
|---|---|---|
| C1 | The diffusion model places hidden players closer than the best classical method | position-weighted median ratio diffusion / best classical < 1 and its 95% interval is below 1 |
| C2 | It is at least as accurate as the transformer | position-weighted median ratio diffusion / transformer ≤ 1.0 (report the interval either way) |
| C3 | Its 90% regions are calibrated at every gap length | pooled coverage in 85–95% in every gap bucket (the resume target 89–91% is reported, not a rule) |
| C4 | It gives back more pitch control than the classical methods | agreement pooled over all scored frames above `kalman`, `spline` and `centroid_vote`, and the 95% interval of each difference above 0 |
| C5 | Per match, it beats the best classical method | count of the 10 matches where its median is lower (reported as k of 10) |

Whatever the result, it is reported, including failed claims. The transformer and the diffusion model are both
reported; if the diffusion model loses to the transformer, the docs say so.

## 8. After cross-validation (fixed now)
- **Released models:** train both models on all 10 matches, for the median best epoch over the 10 folds, no early
  stopping; transformer std scale = median of the 10 fold scales; diffusion eta = the most frequent fold choice.
  They replace `data/models/05-more-data.pt` and the 2.70 m headline (old results stay in git history and in specs 05–13).
- **Browser:** the diffusion denoiser is exported to ONNX and sampled with 10 DDIM steps. It ships on the site only if one
  demo clip with K = 20 samples runs in under 0.5 s (median of 5 runs, `scripts/bench_web.mjs`, desktop Chromium);
  otherwise the site keeps the transformer and the diffusion numbers are reported offline.

## 9. Procedure and changes
- Code: `src/pitchvision/diffusion.py`, `src/pitchvision/centroid_vote.py`, `scripts/lomo_fold.py` (one fold, writes
  `results/15-lomo/fold-<match>.json` and a per-cell `.npz`), `scripts/lomo_summary.py` (all folds → `results/15-lomo.json`).
  Runs on Kaggle GPU (`kaggle/lomo/`), all folds from one code commit.
- A fold that fails for an infrastructure reason (time-out, download) is rerun with the same code. Any code change after
  the first fold result is a dated amendment below, with the reason, and applies to all folds (re-run all).

## Amendment A1 (2026-11-24, before any fold was run)
A local check before the GPU runs (diffusion model trained for 4 epochs on Metrica 1, sampled on 50 windows of Metrica 2;
development only, not a result) showed that the noise-prediction model trained well (denoising loss 0.69 to 0.09) but its
samples were far off (spread of about 50 m): with this schedule the last step has ᾱ = 3e-5, so the first DDIM steps
divide the noise error by √ᾱ ≈ 0.006 and the later steps cannot undo it. Two changes, both standard and fixed before any
fold run:
- The denoiser predicts **v = √ᾱ·noise − √(1−ᾱ)·x₀** (Salimans and Ho 2022) instead of the noise, and the loss is the MSE
  on v. The sampler gets x₀ = √ᾱ·x − √(1−ᾱ)·v, which stays bounded at every step. Schedule, steps, K and the rest unchanged.
- The x₀ estimate inside the sampler is clipped to ±25 residual units (±125 m), more than the pitch, as a guard.

The same local check after the change: sample-mean median error 2.49 m (Kalman anchor on the same cells 1.98 m, after only
4 epochs on one match), spread 3.6 m, coverage 81.5%, so the sampler is sound. These development numbers are not results.

## Amendment A2 (2026-11-24, owner's decision, before any fold was run)
GPU quota is shared with other projects, so the cross-validation runs **5 folds instead of 10** (about 15 GPU hours).
- **Held-out matches, fixed by rule now (not by any result):** every other match of the sorted list of section 2,
  starting with the first: `idsse-J03WMX`, `idsse-J03WOH`, `idsse-J03WPY`, `idsse-J03WR9`, `metrica-2`. That is 4 IDSSE
  and 1 Metrica match, so both providers are tested. (`pitchvision.lomo.TEST_MATCHES`.)
- **Each fold still uses the other 9 matches:** 8 for training and 1 for tuning, chosen as in section 2 (the next match
  in the list, skipping J03WN1). All 10 matches stay in the training pool.
- **Pooled numbers** are over the 5 held-out matches: match-weighted = mean of 5 per-match values; position-weighted =
  all cells of the 5 folds; the stratified bootstrap has all 5 matches in every sample. C5 is reported as k of 5.
- **Released models (section 8)** are a separate, final training job on all 10 matches, with the median best epoch,
  median std scale and most frequent eta over the 5 folds.
- Wherever these numbers are quoted (README, site, resume), they are called "5-fold leave-one-match-out
  cross-validation over a 10-match pool", not 10 folds and not a locked test.

## Result (2026-11-28)

All 5 folds of amendment A2 ran from one code commit (`8f82af3`). Pooled by `scripts/lomo_summary.py` into
`results/15-lomo.json` (block bootstrap, 5-minute blocks, 2,000 resamples). Per-fold summaries: `results/15-lomo/<match>/`.
**This is 5-fold leave-one-match-out cross-validation over a 10-match pool, not a fresh locked test.**

| Method | Median error, m (position-weighted, 95% CI) | Mean error, m | Energy score, m | 90% coverage | Pitch-control agreement |
|---|---|---|---|---|---|
| Spline | 2.72 [2.50, 3.00] | 9.28 | n/a | n/a | 0.927 |
| Kalman | 3.18 [2.94, 3.47] | 9.23 | 8.67 | 0.895 | 0.920 |
| Centroid vote (online) | 8.77 [8.42, 9.19] | 12.05 | n/a | n/a | 0.862 |
| Transformer (v1) | **2.38 [2.24, 2.55]** | **4.51** | **3.27** | 0.916 | **0.945** |
| Diffusion (v2, sample mean) | 2.62 [2.47, 2.80] | 4.83 | 3.51 | 0.829 | 0.940 |

Visible-only pitch-control agreement 0.724 (12,434 frames). Per fold, the transformer has the lowest median in 3 of 5
and the spline in 2 of 5 (folds 3 and 5, the two matches with the shortest gaps); the transformer has the lowest
mean error, energy score and agreement in all 5.

Diffusion 90% coverage by gap: <1 s 0.915, 1–3 s 0.953, 3–6 s 0.927, 6–9.6 s 0.877, ≥9.6 s 0.799 (74% of scored
cells are in the last bucket). Sample spread is positive and rises with the gap (0.16 m to 4.36 m), so the spread
check passes; the spread is too small at long gaps.

| Claim | Rule | Result |
|---|---|---|
| C1 beats the best classical method (spline) | ratio < 1, CI below 1 | **holds**: 0.963 [0.929, 0.996] |
| C2 at least as accurate as the transformer | ratio ≤ 1.0 | **fails**: 1.101 [1.089, 1.108] (10% worse) |
| C3 calibrated in every gap bucket | 85–95% in each | **fails**: 0.799 at ≥9.6 s, 0.953 at 1–3 s |
| C4 more pitch control than the classical methods | above each, CI of difference > 0 | **holds** |
| C5 per match beats the best classical method | k of 5 | 3 of 5 |

Decision: the diffusion model beats the classical methods but **does not beat the transformer**, and its 90% regions
are too narrow at long gaps. As section 7 states, the docs say so; the transformer stays the main model. Likely
causes (not tested here): the point estimate is a mean of 20 samples rather than a direct regression, the sampler
chose eta = 0 (the least spread), and the diffusion model has no fitted spread scale, while the transformer has one
(pooled released setting 1.13).

Section 8 (released all-10-match models, ONNX timing) has not been run yet. Until it is, the README headline stays
spec 05's.
