# pitchvision

**See the players the camera misses.** A TV broadcast of a football match shows only a part of the pitch.
pitchvision gives the position of each hidden player, with a 90% region for where the player probably is.
It also predicts where each player will be in the next 4 seconds.

**Live:** <https://shlokkvaishnav.github.io/pitchvision/> — the model runs in your browser.

[![The pitchvision site](docs/site.png)](https://shlokkvaishnav.github.io/pitchvision/)

## What it does
- **Imputation.** For each hidden player, the model gives a position and an uncertainty (a 2-D Gaussian).
- **Forecast.** The same model predicts the positions of all 22 players 4 seconds ahead.
- **Realistic camera masks.** Which players are hidden comes from real broadcast camera views (SkillCorner), laid onto full-pitch tracking data. Not random masks.
  In our data, about half of the player positions are off-camera, and 74% of the hidden time is in gaps of 9.6 s or more.
- **A web page with no server.** The model is exported to ONNX. The page runs it with onnxruntime-web on real match clips.

## How the model works
A Kalman smoother gives a first estimate from the motion of each player before and after the gap.
A transformer looks at all 22 players and the ball together, and it learns a correction to that estimate.
The last layer starts at zero, thus before training the model is equal to the Kalman estimate.

```mermaid
flowchart LR
  T[Tracking data<br/>Metrica, IDSSE] --> M[Camera mask<br/>SkillCorner views]
  M --> K[Kalman smoother<br/>first estimate + variance]
  K --> X[Transformer<br/>correction, all 22 players]
  M --> X
  X --> C[Std scale<br/>fitted on a tuning match]
  C --> O[ONNX export] --> W[site/<br/>onnxruntime-web]
```

## What works and what does not
Final test: 2 locked Bundesliga matches that were never loaded during development (IDSSE, 2.9 million hidden player positions).
The model, its settings and the rules were fixed and committed before the run. It was run once.

| Part | Result on the locked matches |
|---|---|
| Imputation | **Works.** Median error 2.70 m, against 3.28 m for the best classical method (cubic spline) and 3.81 m for a Kalman smoother. Mean error 4.78 m, against 9.32 m for the spline. |
| Uncertainty | **Works.** The 90% regions contain the true position 92.4% of the time. |
| Forecast at 4 s | **Works.** Mean error 5.18 m, against 6.97 m for constant velocity (on the 66% of positions where constant velocity can make a guess). |
| Smooth motion | **Fails.** The forecast wobbles frame to frame. A simple moving average removes most of it at no cost in accuracy, but it's still jerkier than real players. |
| Red cards | **Not handled.** The model needs 11 v 11. One locked match had a red card, so only 7% of it could be tested; on that part the model only tied the spline. |
| With 2 training matches | **Did not work.** The model was worse than the Kalman smoother. With 5 matches it is better. |
| "What if" plays | **Not built.** |

Caveats: the input is tracking data with broadcast camera views laid on top as masks; there is no video detection.
Two test matches from one league and one training seed is a small sample.
Every experiment has a plan with pass/fail rules written before the run, in [`specs/`](specs/). Failed tests stay in.

## Engineering
- **The browser and Python agree.** A Node test runs the ONNX model in onnxruntime-web and compares it with PyTorch on the demo clips (difference < 1e-4).
- **Fast data loading.** A vectorised path for kloppy loads one IDSSE match in 7 s instead of 121 s.
- **GPU training on Kaggle.** Each run has its own kernel and its own source dataset. Results are JSON files with the git commit.
- **Checks on each push:** ruff (lint), mypy (types), pytest, Node tests. CI runs only on push, never on a schedule.
- **No raw data in the repository.** The site uses 3 short clips of open IDSSE data (CC BY 4.0), with attribution.

## Quick start
```bash
make install        # Python 3.10+, creates .venv
npm ci              # onnxruntime-web, for the browser parity test
make check          # lint, types, tests
```
Train: `PYTHONPATH=src python scripts/train_more.py --out results` (a GPU is recommended; about 1 hour on a Kaggle GPU).

## Layout
| Path | Content |
|---|---|
| `src/pitchvision/` | Data loaders, camera masks, baselines (Kalman, spline), windows, model, calibration |
| `scripts/` | One script per experiment, plus the demo export |
| `kaggle/` | Kaggle kernels for GPU and CPU runs |
| `specs/` | The plan and the result of each experiment |
| `results/` | The result of each experiment (JSON, with the git commit) |
| `site/` | The web page (HTML, CSS, JavaScript, no framework) |

## Data and credits
- IDSSE Bundesliga tracking data: Bassek et al., Scientific Data 12, 195 (2025), CC BY 4.0.
- [Metrica Sports sample data](https://github.com/metrica-sports/sample-data).
- [SkillCorner open data](https://github.com/SkillCorner/opendata): broadcast camera views for the masks.
- [kloppy](https://github.com/PySport/kloppy) for data loading.
- Fonts: Literata and Instrument Sans (SIL Open Font License), in `site/assets/licenses/`.
