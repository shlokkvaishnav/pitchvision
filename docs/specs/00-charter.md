# Spec 00 — Project charter

Status: We committed this file before we downloaded data.
We change this file only with a dated amendment at the end.

## 1. Problem
Broadcast tracking data comes from TV video. It is cheap, and it is available for many matches.
But the TV camera shows only a part of the pitch. Thus, many players are missing at each time step.

Full-pitch tracking data shows all players. But it is expensive, and most of it is not open.
Open research uses a small number of matches.

## 2. Objective
We make one masked generative model of all 22 players and the ball.
The model fills in the missing cells of a table of players × time steps.
It does two tasks:

1. **Imputation.** The model fills in the players that the camera does not show.
2. **Forecast.** The model predicts the future positions of all players.

We then use the two tasks together:

1. The input is broadcast tracking data. Some players are missing.
2. Imputation gives a full 22-player state, with an uncertainty for each player.
3. The forecast gives the real future and "what if" futures (for example: a pass to the left, a higher press).
4. The value of a decision is the difference in threat between the real future and the "what if" future.

## 3. Data
| Source | Content | Use |
|---|---|---|
| PFF FC World Cup 2022 | 64 matches. Full-pitch tracking and events. | Train and test. Has ground truth for hidden players. |
| Metrica Sports sample data | 3 matches. Full-pitch tracking and events. | Train and test. |
| SkillCorner open data | 10 A-League matches. Broadcast tracking. MIT license. | Calibrate the camera mask. Real broadcast checks. |

We read all sources with kloppy.
We keep raw data in `data/`. Git ignores this folder. Do not commit raw data. Do not publish raw data.
The tests use only synthetic data.

## 4. Splits
We split the data by match.
A fixed hash of the match ID puts 20% of the full-pitch matches in the **holdout**.

The holdout is locked. Only the final spec can use it.
The code enforces the lock: `pitchvision.splits.assert_not_holdout()` raises an error.

SkillCorner matches have no ground truth for hidden players.
Thus, we do not use them for rules that need ground truth.

## 5. Camera mask
Full-pitch data does not have missing players. Thus, we simulate the broadcast view.

- The camera footprint on the pitch is a trapezoid.
- The trapezoid follows the ball along the length of the pitch. The pan is smooth.
- The far side of the trapezoid is wider than the near side, because of perspective.
- A player outside the trapezoid is hidden.

Spec 01 sets the sizes of the trapezoid.
We measure the fraction of visible players per frame in the SkillCorner data.
Then we set the sizes so that the simulated fraction is the same.

## 6. Final rules (holdout)
We set these rules now, before the experiments.
We can change them only with an amendment, and only before we use the holdout.

| Rule | Pass condition |
|---|---|
| R1 Imputation | The median position error on hidden players is at least 10% lower than the best baseline. Baselines: constant velocity, Kalman smoother, formation template. |
| R2 Calibration | The 90% prediction regions contain the true position for 85% to 95% of hidden player-frames. |
| R3 Forecast | The average position error at 4 s is lower than the best baseline. The fraction of frames with impossible motion (speed > 12 m/s or acceleration > 10 m/s²) is not more than in the true data. |
| R4 Pipeline stability | We answer each "what if" question two times: from the true input and from the imputed input. The two answers have the same sign in at least 80% of the questions. The median absolute difference is less than 25% of the median absolute value. |

If a rule fails, we keep the result and we report it.

## 7. Rules for each experiment
1. Write a spec in `specs/NN-name.md`. Commit it before the code that runs it.
2. Give each spec its pass/fail rules, with numbers, before the run.
3. Write the results to `results/NN-name.json`, with the git commit SHA.
4. Keep the results of failed rules. Do not remove them.
5. Record each change after a run as an amendment (A1, A2, …), with a date and a reason.

## 8. Output
A static site on GitHub Pages. It has no server.
The site shows statistics, charts and short replays. It does not show raw data.
