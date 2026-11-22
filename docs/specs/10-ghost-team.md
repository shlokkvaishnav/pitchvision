# Spec 10 — Ghost team: where would these players usually stand?

Status: registered on 2026-11-22, before any ghost model was trained or any data was cut for it.

## 1. Objective
For one team (the "ghosted" team), predict where each of its 11 players would stand in this moment, given the other team,
the ball and each player's usual spot. The difference between the real player and the ghost is a "positioning gap": how
far a player is from where a typical Bundesliga or Metrica side would put that role in this situation. This is a new task;
the imputation model of specs 03 to 08 is not changed.

Closest prior work opened: Le, Yue, Carr and Lucey, "Coordinated Multi-Agent Imitation Learning" (ICML 2017,
arXiv 1703.03121), which learns defenders' movement from tracking data with latent roles and Kuhn–Munkres role assignment.
The SSAC 2017 "Data-Driven Ghosting" paper by the same group could not be opened (HTTP 404) and is not cited.

## 2. Inputs and output (per frame, 10 Hz, both teams 11 v 11 with all positions known)
- Other team: position and velocity (over the last 0.5 s) of its 11 players. The ball: position and velocity.
- Ghosted team: for each player, the **role**: their mean position over that half of the match (teams change ends).
  The model never sees the ghosted players' positions in that frame.
- Output: a position and a 2-D spread (log std) per ghosted player; loss: Gaussian NLL, as `pitchvision.model.gaussian_nll`.
- Each frame gives two samples: home ghosted and away ghosted.
- Frames without 22 players (red cards, missing data) are left out.

## 3. Models
| Name | Prediction for a ghosted player | Role |
|---|---|---|
| `role` | Their role position. | Baseline. |
| `linear_shift` | Role + a linear function of (ball position, ball velocity, role, other team's centroid), fitted on training frames. | Baseline: the team slides with the ball. |
| `ghost` | Transformer encoder over 23 tokens (11 other-team players, the ball, 11 ghost queries with their role), 4 layers, width 128, 4 heads, reusing `AxialBlock`'s attention style but over players only. | Candidate. |

## 4. Data and splits
- Development: Metrica games 1 to 3 and IDSSE `J03WMX J03WOY J03WPY J03WQQ J03WR9` (8 matches).
  Model and settings chosen by leave-one-match-out over these 8 (mean error).
- Test, run once: IDSSE `J03WOH` and `J03WN1`, the spec 07 matches. They were the imputation test, but no model has been
  trained or tuned on them for this task. They are loaded with `allow_holdout=True` only by `scripts/ghost.py --test`.

## 5. Metrics and rules (test matches, all frames that qualify, both teams ghosted)
- Mean and median distance between the ghost and the real player (m); coverage of the 90% ellipse.
- Interval: block bootstrap over 5-minute stretches (10,000 samples, seed 0), as spec 07.

| Rule | Pass condition |
|---|---|
| G1 | Mean error of `ghost` ≤ 0.90 × mean error of `linear_shift`, and the 95% interval of the ratio is below 1. |
| G2 | 90% ellipse coverage of `ghost` between 85% and 95%. |

## 6. Output
`src/pitchvision/ghost.py` (features, baselines, model), `scripts/ghost.py` (development and `--test`),
`results/10-ghost-dev.json`, `results/10-ghost-test.json`. Site: ghost rings for the chosen team, with the gap per player.
