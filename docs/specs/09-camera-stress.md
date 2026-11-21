# Spec 09 — Camera stress test (exploratory)

Status: committed before the first run. **Exploratory only.** The locked holdout was used once (spec 07), so this spec
has no pass/fail rule, makes no new headline claim, and does not change the model, the scale factor or the site.

## 1. Objective
All results so far use one kind of camera mask: real SkillCorner camera views, replayed so that they follow the ball
(spec 01 A1, the replay_ball mask). A real broadcast also zooms, lags behind play, and cuts away to replays.
This spec measures how the spec 05 model and its Kalman baseline behave when the camera mask is changed in these ways.
It also reports 90% coverage by gap length and by player role, which earlier specs only reported pooled.

## 2. Data
- The spec 05 test matches: `idsse-J03WQQ` (donor 1959846) and `idsse-J03WR9` (donor 1986691). No model was trained on them.
  They are development matches; spec 08 already used them.
- **The locked holdout matches `idsse-J03WN1` and `idsse-J03WOH` are not loaded.** The script asserts it, and the loader
  refuses them.
- 10 Hz, 9 s windows, stride 2 s, as spec 05. The windows do not depend on the mask, so every condition has the same windows.

## 3. Model and baselines (fixed)
- Model: `data/models/05-more-data.pt` (spec 05, seed 0, the site model), std scale s = 1.1747. No training, no tuning.
- The Kalman anchor and all baselines are recomputed from the perturbed mask. The model never sees a hidden position.
- Kalman baseline: the RTS smoother of spec 02 over the whole match (`baselines.kalman_smoother`). Its 90% region:
  the smoother variance as used by the anchor (the anchor coverage of spec 05).
- Cubic spline (spec 06 bar), for reference.

## 4. Perturbations of the camera footprint
The footprint is the (T, 4, 2) camera polygon of the replay_ball mask. A player is hidden when outside it.

| Condition | Change |
|---|---|
| `none` | No change (the spec 05 setting; a check that the script reproduces spec 05). |
| `zoom_0.7`, `zoom_0.85`, `zoom_1.15`, `zoom_1.3` | Scale each frame's polygon about its centroid (mean of the 4 corners) by this factor. Below 1: a tighter view. |
| `lag_0.5s`, `lag_1s`, `lag_2s` | Delayed pan: the polygon at frame t is the polygon of frame t − lag. At the start of a period, the first polygon of the period. |
| `cut_2s`, `cut_4s`, `cut_8s` | Replay cuts: no player is visible during a cut. Cut starts: a Poisson process, on average one cut per 60 s, inside each period, seed fixed per match; the same starts for every cut length. The ball stays observed. |
| `cut_4s_ball` | As `cut_4s`, but the ball is also hidden during the cuts (closer to a real replay; the model never saw a hidden ball in training). |

## 5. Metrics (per condition, both matches pooled)
- Share of present player positions that are hidden, and the number of hidden player cells.
- Hidden player cells where the Kalman smoother has an estimate: median and mean error of the model, the Kalman smoother
  and the spline (same cells within a condition). The ratio model / Kalman of the medians.
- 90% coverage: the model (after s) and the Kalman anchor.
- By gap length (the hidden run in the full match that contains the cell, one period at a time; spec 06 bins
  < 1 s, 1–3 s, 3–6 s, 6–9.6 s, ≥ 9.6 s) and by player role (from kloppy's starting position: goalkeeper, defender,
  midfielder, forward; `unknown` if missing): the share of cells, the medians of model and Kalman, and both coverages.

Caveats fixed now: the hidden cells differ between conditions, so compare model and Kalman within a condition, not
absolute errors across conditions. Windows overlap, so a cell can count more than once (as in all specs). Two matches,
one league: the numbers describe these matches, not broadcasts in general. No intervals.

## 6. Compute and output
CPU. One match and condition takes about one minute (model about 45 s for 1,778 windows), so all 12 conditions on both
matches fit on a laptop; no subset. `scripts/camera_stress.py` writes `results/09-camera-stress.json` with the git commit.

## Result (2026-11-21, laptop CPU, commit 8abfdaa)
`results/09-camera-stress.json`. Exploratory: no rule, no new claim. The `none` row reproduces spec 05 exactly
(2.43 m, Kalman 3.27 m, spline 2.82 m, coverage 92.1%, anchor 88.8%), so the script matches the spec 05 evaluation.

| Condition | Hidden | Hidden cells | Model median | Kalman median | Spline median | Model / Kalman | Mean, model / Kalman | Coverage, model | Coverage, Kalman |
|---|---|---|---|---|---|---|---|---|---|
| `none` | 48.8% | 4,554,291 | **2.43 m** | 3.27 m | 2.82 m | 0.74 | 4.58 / 9.66 m | 92.1% | 88.8% |
| `zoom_0.7` | 67.0% | 6,257,015 | **3.77 m** | 5.53 m | 5.00 m | 0.68 | 6.63 / 13.66 m | 89.5% | 88.4% |
| `zoom_0.85` | 57.1% | 5,324,286 | **2.93 m** | 4.13 m | 3.65 m | 0.71 | 5.34 / 10.98 m | 91.0% | 88.3% |
| `zoom_1.15` | 41.5% | 3,881,945 | **2.11 m** | 2.74 m | 2.30 m | 0.77 | 4.00 / 8.77 m | 93.2% | 90.0% |
| `zoom_1.3` | 35.2% | 3,291,015 | **1.90 m** | 2.39 m | 1.96 m | 0.79 | 3.63 / 7.98 m | 94.1% | 91.4% |
| `lag_0.5s` | 48.7% | 4,546,561 | **2.48 m** | 3.34 m | 2.87 m | 0.74 | 4.60 / 9.48 m | 91.9% | 88.7% |
| `lag_1s` | 48.7% | 4,545,582 | **2.53 m** | 3.35 m | 2.88 m | 0.75 | 4.65 / 9.46 m | 91.5% | 88.9% |
| `lag_2s` | 49.1% | 4,575,980 | **2.54 m** | 3.32 m | 2.90 m | 0.77 | 4.65 / 9.62 m | 90.9% | 89.0% |
| `cut_2s` | 50.3% | 4,692,399 | **2.33 m** | 3.10 m | 2.64 m | 0.75 | 4.54 / 9.74 m | 92.1% | 88.8% |
| `cut_4s` | 51.8% | 4,830,827 | **2.34 m** | 3.06 m | 2.60 m | 0.76 | 4.53 / 9.60 m | 91.8% | 88.4% |
| `cut_8s` | 54.8% | 5,109,420 | **2.62 m** | 3.39 m | 2.92 m | 0.77 | 4.77 / 9.76 m | 91.3% | 87.7% |
| `cut_4s_ball` | 51.8% | 4,830,827 | **2.35 m** | 3.06 m | 2.60 m | 0.77 | 4.54 / 9.60 m | 91.5% | 88.4% |

**By gap length and role, `none` (the spec 05 setting).** Coverage was only reported pooled before.

| Group | Share of cells | Model median | Kalman median | Coverage, model | Coverage, Kalman |
|---|---|---|---|---|---|
| < 1 s | 0.9% | 0.08 m | 0.06 m | 94.9% | 96.9% |
| 1–3 s | 5.2% | 0.17 m | 0.16 m | 91.5% | 92.5% |
| 3–6 s | 8.7% | 0.46 m | 0.45 m | 90.8% | 87.7% |
| 6–9.6 s | 9.8% | 0.94 m | 0.97 m | 92.4% | 84.4% |
| ≥ 9.6 s | 75.4% | 3.68 m | 5.80 m | 92.2% | 89.2% |
| Goalkeeper | 12.8% | 2.82 m | 7.36 m | 91.7% | 96.9% |
| Defender | 30.0% | 2.67 m | 3.54 m | 92.9% | 89.2% |
| Midfielder | 32.1% | 2.36 m | 2.81 m | 92.3% | 87.8% |
| Forward | 14.1% | 2.19 m | 2.56 m | 91.3% | 86.9% |
| Unknown | 11.0% | 1.84 m | 2.06 m | 90.8% | 83.9% |

The same tables for every condition are in the JSON.

**What we see (descriptive, two matches).**
- **Zoom matters most.** A tighter view (0.7) hides 67% of positions instead of 49%; the model error rises to 3.77 m and
  coverage falls to 89.5%, still inside the 85–95% band of spec 05. The model's lead over the Kalman smoother and the
  spline grows as the view gets tighter (ratio 0.68 at 0.7) and shrinks as it gets wider (0.79 at 1.3, where it is
  close to the spline: 1.90 vs 1.96 m).
- **Pan lag up to 2 s changes little:** +0.11 m median, coverage 90.9%.
- **Replay cuts** of 2 or 4 s lower the median. They add many short, easy hidden cells next to visible ones, so the
  pooled cells change; this is not a real improvement. 8 s cuts raise the median to 2.62 m. Hiding the ball as well
  during 4 s cuts (`cut_4s_ball`) changes almost nothing (2.35 vs 2.34 m, coverage 91.5%).
- **Calibration by group.** The model's coverage stays between 90.8% and 94.9% in every gap bin and role, while the
  Kalman smoother's ranges from 83.9% to 96.9%. The model gains almost nothing over the Kalman smoother in gaps under
  9.6 s; nearly all of its gain is in the long gaps (75% of the cells).
- **Goalkeepers** are where the Kalman smoother is worst (7.36 m) and the model gains most (2.82 m).
- `unknown` is 11% of cells: players whose starting position kloppy does not give (mostly substitutes).

**Limits.** Two development matches from one league, synthetic perturbations of transplanted camera views, no intervals.
The perturbations are simple: real zoom changes over time, and real replays also repeat earlier footage.

## Note: smaller ONNX files (2026-11-21, exploratory)
Added after the camera run; not part of the plan above, no rule. `scripts/onnx_variants.py` converts the site model to
fp16 (onnxconverter-common, inputs and outputs kept fp32, Cast nodes kept) and int8 (onnxruntime dynamic quantisation of
the weights), scores each on the spec 05 test windows (4,679 windows, 4,554,291 hidden cells, match-level anchors,
onnxruntime 1.32 CPU), and times onnxruntime-web 1.31.1 (wasm, Node) on the 3 demo clips.
`results/09-onnx-variants.json` (its commit field is the parent commit; the script is unchanged since).

| Variant | Size | Median error | Mean error | Coverage | Max mean change vs fp32 | Web (wasm) per clip |
|---|---|---|---|---|---|---|
| fp32 (site) | 4.40 MB | 2.433 m | 4.582 m | 92.08% | — | 93 ms |
| fp16 | 2.30 MB | 2.433 m | 4.581 m | 92.08% | 0.16 m | 94 ms |
| int8 | 1.32 MB | 2.441 m | 4.591 m | 91.45% | 2.49 m | 76 ms |

- fp16 halves the file with no change in accuracy, but is not faster on wasm (the CPU kernels compute in fp32).
- int8 is 3.3x smaller and about 20% faster, at +0.008 m median and −0.6 points coverage, with some single cells moving
  by up to 2.5 m. The site keeps fp32; fp16 is the safe choice if the download size matters.
- Timings are medians of 60 runs on one laptop, rough; WebGPU was not tested.
