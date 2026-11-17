# Spec 02 — Imputation baselines on full-pitch data

Status: We committed this file before the baseline run.

## 1. Objective
1. Measure the error of each baseline on hidden players, with the camera mask from spec 01.
2. Select the best baseline. This baseline is the bar for rule R1 (spec 00).
3. Show how the error changes with the length of the gap.

## 2. Data
- Metrica Sports sample games 1, 2 and 3. All three are development matches (spec 00, section 4).
- PFF FC World Cup 2022 is not used here, because we do not have the files yet. Spec 03 adds it.

## 3. Frame rate
The project uses **10 Hz** for all data. This is the SkillCorner rate.
Metrica (25 Hz) is resampled with linear interpolation in time.
A frame gets NaN if one of the two source frames around it has NaN.

## 4. Mask
- We use the trapezoid with the parameters in `results/01-camera.json` (`final_params_all_matches`).
- If spec 01 rule C1 or C2 failed, we use the real SkillCorner footprints. In that case, we write an amendment first.
- The camera pan restarts at the start of each period.

## 5. Baselines
| Name | Uses future frames | Note |
|---|---|---|
| hold_last | no | Last visible position |
| constant_velocity | no | Last visible velocity, speed capped at 8 m/s |
| linear_interpolation | yes | Straight line across the gap |
| formation_template | no | Centroid of the other visible team-mates + mean offset. Offsets come from the visible frames of the same match. |
| kalman_smoother | yes | Constant-velocity RTS smoother |

Imputation is offline. Thus, a baseline can use frames after the gap.

## 6. Metrics
- Position error in metres on hidden player-frames: median, mean, 90th percentile.
- The same metrics for each gap-length bin: < 1 s, 1 s to 3 s, 3 s to 10 s, > 10 s.
- The fraction of hidden player-frames. We compare it with the fraction of SkillCorner player-frames with `is_detected = False`.

## 7. Rules
| Rule | Condition |
|---|---|
| B1 Bar for R1 | The baseline with the lowest pooled median error over the three games is the bar for rule R1. This is a choice, not a pass/fail rule. |
| B2 Pipeline check | Linear interpolation has a median error of less than 1.0 m on gaps shorter than 1 s. If it fails, the pipeline has a defect, and we stop. |

## 8. Output
`results/02-baselines.json`: the metrics for each game, each baseline and each gap bin, the selected bar, and the git commit SHA.

## Amendment A1 (2026-11-15)
**Reason.** Spec 01 rules C1 and C2 failed for the trapezoid (spec 01, A1), and amendment A2 of spec 01 selected `replay_ball`.
No candidate passed all of Q1 to Q3. `replay_ball` passed Q1 (hidden fraction 46.0%, real 43.2%).
It failed Q2 (gap-length KS 0.225) and Q3 (it hides too many players near the near touchline: 67%, real 40%).

**Change to section 4.**
- The mask is `replay_ball`. The donors are the SkillCorner matches in the sorted list of IDs: the first for Metrica game 1, the second for game 2, the third for game 3.
- The rule check on spec 01 C1 and C2 is removed from the script.
- The report gives the Q2 and Q3 failures as a limit of all results that use this mask.

## Amendment A2 (2026-11-15)
**Result of run 1** (kept in `results/02-baselines-run1.json`). B2 passed. The bar was `kalman_smoother` (pooled median 2.68 m).
`formation_template` had a median error of 27 m, also on gaps shorter than 1 s.

**Cause.** A defect in the baseline. The teams change ends at half-time (Metrica game 1: the home goalkeeper has a mean x of −39 m in period 1 and +43 m in period 2).
The mean offset over the full match is thus near zero.

**Change.** `formation_template` computes the offsets separately for each period (`formation_template_by_period`).
This is a defect fix, not a change of method. All other baselines and the rules do not change. Run 2 replaces `results/02-baselines.json`.

## Result note (2026-11-15, run 2)
B2 passes (linear interpolation, gaps < 1 s: 0.03 m to 0.04 m). The bar for R1 is `kalman_smoother`, pooled median **2.68 m**.
Pooled medians: kalman_smoother 2.68 m, linear_interpolation 3.33 m, hold_last 7.87 m, constant_velocity 8.18 m, formation_template 20.91 m.
After the fix, `formation_template` is still weak (13 m on gaps < 1 s). This is a limit of the method, not a defect:
the camera shows one part of the pitch, so the visible team-mates are a biased sample of the team.
Metrica hidden fraction with the replay mask: 49.4%. SkillCorner real undetected fraction: 43.3%.
