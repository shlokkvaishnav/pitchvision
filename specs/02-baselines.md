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
