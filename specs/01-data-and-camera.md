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
