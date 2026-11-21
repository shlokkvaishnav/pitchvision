# Method and results

This page is the one place to read how pitchvision was tested. The detailed plans live in [`docs/specs/`](specs/).

## How the work was done
- **A written plan before every experiment.** Each spec lists the data, the model and pass/fail rules with numbers.
  It was committed before the first run. Changes after that are dated amendments.
- **A locked final test.** A fixed hash of the match ID put 20% of the matches in a holdout. The loaders refuse those
  matches (`pitchvision.splits`) unless the final spec asks for them. They were used once, at the end ([spec 07](specs/07-holdout.md)).
- **Failures stay in.** Failed rules, errata and corrections are kept in the specs.
- **Every number has a source.** Each result is a JSON file in `results/` with the git commit that made it.

## The setup
- **Input:** full-pitch tracking data (positions of all 22 players and the ball, resampled to 10 Hz).
- **Camera masks:** a broadcast camera shows only part of the pitch. SkillCorner open data gives the real camera view of
  20 broadcast matches. We lay those camera views over the full-pitch matches, frame by frame, following the ball
  ("replay mask", [spec 01](specs/01-data-and-camera.md)). A player outside the view is hidden. This is not video processing:
  there is no detection from pixels.
- **Tasks:** fill in hidden players inside a 9 s window (imputation, with a 90% region), and predict all players 4 s ahead (forecast).
- **Model:** a Kalman smoother over the whole match gives a first estimate and its variance. An axial transformer
  (attention over time and over players) learns a correction to the mean and the log-standard deviation.
  The output layer starts at zero, so the untrained model equals the Kalman estimate. One scale factor on the standard
  deviation is fitted on a separate tuning match.

## All experiments
| Spec | Question | Result |
|---|---|---|
| [00](specs/00-charter.md) | Charter: goals, splits, final rules | — |
| [01](specs/01-data-and-camera.md) | Can we build realistic camera masks? | Pre-registered rules failed. We use the best candidate (replay of real camera views): about half of player positions hidden. |
| [02](specs/02-baselines.md) | Classical baselines on Metrica | Kalman smoother best: 2.68 m median. |
| [03](specs/03-masked-model.md) | Transformer alone (version A) | Imputation fails (5.39 m vs 2.13 m). Uncertainty passes. Forecast passes after an erratum (7.42 m vs 8.62 m). |
| [04](specs/04-anchored-model.md) | Learn a correction to the Kalman estimate (version B), 2 training matches | Forecast passes (6.64 m vs 8.62 m). Imputation fails (2.24 m vs 2.13 m). Uncertainty fails (84.1%). |
| [06](specs/06-baseline-audit.md) | Are our baselines strong enough? | A cubic spline beats the Kalman smoother (2.48 m vs 2.92 m): it becomes the bar. 74% of hidden time is in gaps of 9.6 s or more. |
| [05](specs/05-more-data.md) | Version B with 5 training matches, early stopping, calibration | **All pass** on 2 unseen matches: 2.43 m (spline 2.82 m), 92.1% coverage, forecast 5.17 m vs 6.94 m. |
| [07](specs/07-holdout.md) | Final test on the 2 locked matches | Imputation **passes** (2.70 m vs spline 3.28 m). Uncertainty **passes** (92.4%). Forecast error **passes** (5.18 m vs 6.97 m). Smooth motion **fails**. |
| [08](specs/08-seeds-and-intervals.md) | How much do the results move with the seed and with the sample? | 3 seeds: median error 2.37 to 2.43 m, all pass. 95% intervals: error ratio vs spline 0.78 to 0.87 on the locked matches. |
| [09](specs/09-camera-stress.md) | Exploratory, spec 05 test matches: what if the camera zooms, lags or cuts away? | No rule. Tighter view (0.7x) is hardest: 3.77 m (Kalman 5.53 m), coverage 89.5%. Pan lag up to 2 s: +0.11 m. Coverage by gap length and role stays at 90.8 to 94.9%. |

## Related work
- **Graph Imputer** (Omidshafiei et al., "Multiagent off-screen behavior prediction in football", *Scientific Reports*, 2022,
  [doi:10.1038/s41598-022-12547-0](https://doi.org/10.1038/s41598-022-12547-0)), DeepMind and Liverpool FC.
  The closest prior work: it imputes off-screen players with bidirectional variational LSTMs and graph networks, trained on
  105 matches, and uses the result for pitch control. pitchvision solves the same task with a different design: a classical
  Kalman estimate plus a learned correction, so the model can never start worse than the classical method; calibrated 90%
  regions that are checked; only open data; and a final test against the strongest classical baseline we found (a cubic spline).
  We did not run Graph Imputer, so we make no claim about which is more accurate.
- **MIDAS** ([arXiv 2408.10878](https://arxiv.org/abs/2408.10878)) reports that a cubic spline is a strong imputation baseline
  on Metrica. This is why spec 06 added the spline.
- **Choi et al.** ([arXiv 2607.11548](https://arxiv.org/abs/2607.11548)) report that most hidden player observations are in
  long gaps, as we found in spec 06.

## Known limitations
- **The locked result is mostly one match.** J03WOH gives 2.76 million of the 2.93 million scored positions; on J03WN1
  (red card, 7% scorable) the model only tied the spline (2.27 vs 2.30 m).
- **Small data.** 5 training matches, 2 test matches and 2 locked matches, from 2 sources. Results vary little with the training seed (3 seeds, [spec 08](specs/08-seeds-and-intervals.md)), but
  only one seed was run on the locked matches.
  4.6 million hidden positions sounds large, but the independent units are matches.
- **Overlapping windows.** Windows overlap (2 s stride), so a hidden position can be scored more than once. This is the same for every method.
- **Forecast numbers use a subset.** The forecast is compared with constant velocity on the 66 to 68% of positions where constant
  velocity can make a guess. On all positions, the model's 4 s error is about 7.4 m.
- **Jerky forecasts.** The forecast mean wobbles from frame to frame (accelerations above 10 m/s² in about a third of frames).
  A 9-frame moving average removes most of it at no cost in accuracy.
- **11 v 11 only.** Windows need 11 players per team. A red card removes most of a match (one locked match lost 93% of its windows).
- **Transplanted camera views.** The camera views come from other matches, following the ball. A real broadcast of the same match
  would also cut to replays and close-ups. Spec 09 tests simple versions of zoom, pan lag and replay cuts on the spec 05
  test matches (exploratory, not on the locked matches).
- **The "what if" stage of the charter was not built.**
