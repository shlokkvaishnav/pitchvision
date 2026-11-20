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
