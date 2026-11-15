# pitchvision

pitchvision fills in the football players that the TV camera does not show.
Then it predicts the play, and it answers "what if" questions.
One masked generative model does both tasks.

## Why
Broadcast tracking data is cheap, and it is available for many matches.
But the camera shows only a part of the pitch. Thus, many players are missing.
Full-pitch tracking data is expensive, and most of it is not open.

## How it works
1. The input is broadcast tracking data. Some players are missing.
2. The model fills in the missing players. It gives an uncertainty for each player.
3. The model predicts the next seconds of play: the real play and "what if" plays.
4. The difference in threat between the two plays gives the value of a decision.

The model fills in cells of a table of players × time steps.
Imputation hides players. Forecast hides future time steps. Thus, one model can learn both tasks.

## Method
- We write a spec before each experiment. Each spec has pass/fail rules with numbers.
- 20% of the matches are a locked holdout. The code enforces the lock.
- We keep failed results, and we report them.
- We test the full pipeline: do the "what if" answers stay the same when the input is imputed?

Read the charter: [specs/00-charter.md](specs/00-charter.md).

## Milestones
| Milestone | Content | Status |
|---|---|---|
| M0 | Charter, holdout lock, camera mask, CI, site | Done |
| M1 | Data loaders (kloppy), camera mask calibration on SkillCorner data | Next |
| M2 | Baselines: constant velocity, Kalman smoother, formation template | |
| M3 | Masked generative model (imputation and forecast) | |
| M4 | "What if" values and the pipeline stability test | |
| M5 | Final holdout run and report | |

## Quick start
```bash
make install
make check   # lint and tests
```

## Data
PFF FC World Cup 2022, Metrica Sports sample data, and SkillCorner open data.
Do not commit raw data. The site shows statistics only.
