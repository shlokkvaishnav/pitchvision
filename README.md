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

## Results (development data)
| Spec | Test | Result |
|---|---|---|
| [01](specs/01-data-and-camera.md) | A camera mask that hides players as a real broadcast does (SkillCorner, 20 matches) | **Fail** on the pre-registered rules. We use the best candidate: a replay of real camera footprints. It hides 49% of player cells. |
| [02](specs/02-baselines.md) | Baselines on Metrica games 1 to 3 | The Kalman smoother is the bar: 2.68 m median error. Linear interpolation 3.33 m. |
| [03](specs/03-masked-model.md) | Masked model version A | **Calibration passes** (88.4% in the 90% regions). **Imputation fails** (5.39 m against 2.13 m for the Kalman smoother). **Forecast passes** (7.42 m against 8.62 m for constant velocity at 4 s, on the same cells, after an erratum). |
| [04](specs/04-anchored-model.md) | Masked model version B: a learned correction to the Kalman estimate | Running |

The "what if" stage starts only after the imputation passes.

## Quick start
```bash
make install
make check   # lint and tests
```

## Data
PFF FC World Cup 2022, Metrica Sports sample data, and SkillCorner open data.
Do not commit raw data. The site shows statistics only.
