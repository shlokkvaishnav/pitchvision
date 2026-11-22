# Spec 11 — Who receives the pass?

Status: registered on 2026-11-22, before any pass was matched to tracking or any model trained.

## 1. Objective
At the moment a pass is played, predict which teammate receives it, from the positions and velocities of all 22 players
and the ball. On the page, this becomes arrows from the ball carrier with a chance for each teammate, drawn on the
filled-in positions when some players are off-camera.

## 2. Data
- IDSSE development matches `J03WMX J03WOY J03WPY J03WQQ J03WR9` (event and tracking data, Bassek et al. 2025,
  CC BY 4.0). Completed passes with a named receiver. Metrica is not used (its event files do not load the same way).
- The spec 07 holdout matches are not loaded. The result is **exploratory**: leave-one-match-out over the 5 matches.
- Frame: the tracking frame nearest the event time (10 Hz). Check, reported: the share of passes where the passer is
  within 3 m of the ball at that frame. Passes where the passer or the receiver has no position are left out.

## 3. Inputs (per pass)
For each of the 22 players: position and velocity (over 0.5 s), same team as the passer, is the passer, distance to the
passer, distance to the nearest opponent. The ball: position. Coordinates are flipped so the passer's team attacks
towards +x.

## 4. Models (choice among the passer's 10 teammates)
| Name | Score of a teammate | Role |
|---|---|---|
| `nearest` | − distance to the passer. | Baseline. |
| `logit` | Conditional logit (softmax over teammates) on distance to passer, distance to nearest opponent, x gain (receiver x − passer x), and whether the passer→receiver line passes within 2 m of an opponent. Fitted to convergence (full-batch L-BFGS). | Baseline. |
| `gnn` | Transformer encoder over the 22 players + the ball (3 layers, width 64, 4 heads), a score per teammate, softmax. AdamW, learning rate 1e-3, 30 epochs, batch 64, seed 0, cross-entropy. | Candidate. |

## 5. Metrics and rules (pooled over the 5 left-out matches)
Top-1 and top-3 accuracy, log loss. Block bootstrap over 5-minute stretches (10,000 samples, seed 0).

| Rule | Pass condition |
|---|---|
| P1 | Log loss of `gnn` < `logit`, with the 95% interval of the difference below 0. |
| P2 | Top-1 accuracy of `gnn` ≥ `nearest`. |

## 6. Output
`src/pitchvision/passes.py`, `scripts/pass_receiver.py`, `results/11-pass-receiver.json`. Site: arrows with chances.

## Amendment 1 (2026-11-22, after the data check, before any model was scored)
The data check of section 2 failed: at the nearest frame to the event time the passer was within 3 m of the ball in
only 37% to 62% of passes per match, and no constant time shift fixed it (best about 60%). So the frame is now: within
±1.0 s of the event time, the frame where the passer is closest to the ball. Passes where that distance is still more
than 3 m (headers, throw-ins, high balls, tracking gaps) are left out. The share kept per match is reported.

## Result (2026-11-22, commit c49fcda; exploratory, leave-one-match-out)
2,758 passes kept of 3,611 completed passes with a receiver (65% to 88% per match, as printed by the script); after snapping, the median
passer-to-ball distance is about 0.5 m. Raw numbers: `results/11-pass-receiver.json`.

| Model | Top-1 | Top-3 | Log loss |
|---|---|---|---|
| nearest | 27.4% | 65.4% | 6.69 |
| logit | 36.6% | 76.3% | **1.76** |
| gnn | **47.1%** | **78.5%** | 2.21 |

- **P1 fails.** The transformer picks the right receiver more often (better in all 5 matches), but its log loss is worse
  than the logit's (difference +0.44, 95% interval +0.33 to +0.56): it is overconfident when wrong. Calibrating it
  (for example a temperature fitted on a separate match) would be a new spec.
- **P2 passes** (47.1% against 27.4% top-1 for the nearest teammate).
