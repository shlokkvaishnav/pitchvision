# Spec 01 — Data loaders and camera mask calibration

Status: We committed this file before the calibration run.

## 1. Objective
1. Load all data sources into one standard format.
2. Fit the sizes of the camera mask (spec 00, section 5) on real broadcast data.
3. Make sure that the mask agrees with the real camera.

## 2. Standard format
All loaders give a `Tracking` object:

| Field | Shape | Content |
|---|---|---|
| `players` | (T, N, 2) | Player positions in metres. The origin is the centre spot. NaN means no data. |
| `ball` | (T, 2) | Ball position in metres. |
| `period` | (T,) | Match period. |
| `player_ids`, `teams` | (N,) | Player ID and team ("home" or "away"). |
| `detected` | (T, N) or none | Broadcast data only. True if the camera showed the player. |
| `corners` | (T, 4, 2) or none | Broadcast data only. The real camera footprint on the pitch. |

The x axis runs along the length of the pitch. The y axis runs across the width.
Negative y is the touchline near the main camera.

## 3. Sources
| Source | Loader | Note |
|---|---|---|
| Metrica Sports sample data | kloppy, coordinates "secondspectrum" | 25 Hz |
| PFF FC World Cup 2022 | kloppy `pff.load_tracking` | The user must get the files with the PFF form. |
| SkillCorner open data | own parser of `tracking_extrapolated.jsonl` | 10 Hz. Gives `is_detected` and `image_corners_projection`. |

The SkillCorner tracking files are in Git LFS. We download them from the GitHub media URL.

## 4. Calibration
SkillCorner gives two things that the real camera did:
- `is_detected`: was the player in the image?
- `image_corners_projection`: the four corners of the image on the pitch.

We fit three parameters of the trapezoid mask: `near_width`, `far_width` and `pan_smoothing`.
The fit target is the agreement between the mask and `is_detected`.

- Grid: `near_width` 20 m to 70 m, step 5 m. `far_width` 40 m to 110 m, step 5 m, and not less than `near_width`.
  `pan_smoothing` in {0, 0.5, 0.8, 0.9, 0.95, 0.98}.
- Frames: periods 1 and 2 only. We use one frame per second for the score.
  The camera pan uses all frames.
- Validation: 5-fold cross-validation by match. We fit on 8 matches and score on the other 2.
- Reference: the mask from the real corners (point in polygon). This is the best result that a footprint can give,
  because a player in the image can be hidden behind another player.

## 5. Rules
| Rule | Pass condition (mean of the 5 validation folds) |
|---|---|
| C1 Agreement | The fitted mask agrees with `is_detected` for at least 85% of player-frames. |
| C2 Near the reference | The agreement is not more than 5 percentage points below the real-corner reference. |
| C3 Count | The mean absolute difference in the number of visible players per frame is not more than 1.0. |

If C1 or C2 fails, spec 02 uses the real SkillCorner footprints (replayed relative to the ball), not the trapezoid.

## 6. Output
`results/01-camera.json`: the fitted parameters, the scores per fold, the reference scores, and the git commit SHA.

## Amendment A1 (2026-11-15)
**Result of the first run.** The trapezoid failed C1, C2 and C3 (agreement 74.6%, reference 97.3%, count error 4.9).
A look at the real footprints of match 1886347 shows two causes:
- The near edge of the image is at y = −30 m (median), not at the near touchline. Thus, players near the near touchline are often not in the image.
- The camera centre follows the ball (correlation 0.965), but it is often 14 m in front of or behind the ball.

**Data.** SkillCorner open data now has 20 matches, not 10. The run used all 20. Each fold fits on 16 matches and scores on 4.

**Decision.** We do not change the trapezoid after we saw the result. Section 5 gave the fallback: replay of the real footprints.

**Replay method.**
1. For each SkillCorner match and period, we keep the footprint sequence. The x coordinates are relative to the ball x. The y coordinates are absolute.
   A frame with no corners keeps the corners of the frame before it.
2. For a target match, the footprint at frame t is the donor footprint at frame t (repeated if the donor is shorter), moved along x by the target ball x.
3. The donor is a different match. For the test below, the donor of a match is the next match in the sorted list of IDs.

**Test of the replay.** For each of the 20 SkillCorner matches, we replay the footprints of its donor on its own ball path.
Then we score the mask against its own `is_detected`, with the same frames as before (one frame per second, periods 1 and 2).
We use the same rules C1, C2 and C3. The reference is again the mask from the real corners of the match.

If the replay also fails, we record the result. Spec 02 then uses the replay as the best available mask, and its report shows this limit.

## Amendment A2 (2026-11-15)
**Result of A1.** The replay relative to the ball also failed C1, C2 and C3 (agreement 66.5%, count error 5.7).
It is worse than the trapezoid (74.6%). Thus, the ball position alone does not give the camera framing at each frame.

**Why the rules change.** Rules C1 to C3 ask for frame-by-frame agreement with the real camera of the same match.
The full-pitch data (Metrica, PFF) has no real camera. The model needs a mask that is realistic as a distribution:
the right fraction of hidden players, the right gap lengths, and the right parts of the pitch.
We keep C1 to C3 as reported numbers. They do not select the mask.

**Candidates.** We set these before the test:
1. `trapezoid`: the fit from the first run (near 35 m, far 40 m, smoothing 0.8).
2. `replay_ball`: the replay of amendment A1.
3. `replay_centroid`: the same replay, but the x coordinates are relative to the mean x of all players on the pitch, not the ball.
   The camera frames the play. The player centroid can show the play better than the ball.

**Test.** The same cross-match scheme as A1 (donor = next match ID), on all 20 SkillCorner matches, one frame per second.
The real mask is `is_detected`.

| Rule | Pass condition (pooled over the 20 matches) |
|---|---|
| Q1 Hidden fraction | Within ±3 percentage points of the real hidden fraction. |
| Q2 Gap lengths | The Kolmogorov-Smirnov distance between the simulated and the real lengths of hidden runs is not more than 0.10. Run lengths come from the 10 Hz frames. |
| Q3 Depth bands | For each band of y (y < −20 m, −20 m to 20 m, y > 20 m), the hidden fraction is within ±5 percentage points of the real one. |

**Selection.** Spec 02 uses the candidate that passes Q1, Q2 and Q3. If more than one passes, we use the one with the highest C1 agreement.
If no candidate passes, we use the candidate with the most passed rules (then the highest C1 agreement), and the reports state this limit.
