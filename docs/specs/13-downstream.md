# Spec 13 — What does filling in hidden players give back to football analysis?

Status: registered on 2026-11-22, before any data was computed for it. Exploratory (development matches only).

## 1. Question
A TV broadcast hides about half the players. Analyses built on tracking data (who controls the space, who will
receive the pass, is the defence in shape) then run on incomplete data. How much accuracy does that cost, and how much
does pitchvision get back by filling the hidden players in? This ties the imputation model (specs 03 to 08) to the
analyses on the page (pitch control, spec 11 passes, spec 10 ghost team).

## 2. Data
- IDSSE J03WQQ and J03WR9: the spec 05 test matches. The released model (`data/models/05-more-data.pt`, std scale
  1.1747) never trained on them. They were its spec 05 test, so this is not a fresh test: **exploratory**.
  The spec 07 holdout matches are not loaded.
- Camera masks: the replay masks of spec 05 (`pitchvision.dataset.donors`, `load_match`).
- Unit: the middle frame (frame 45 of 90) of every window of `load_match`, which gives one frame every 2 s with
  context on both sides; only frames with at least one hidden player.

## 3. Four versions of the 22 positions at each unit
| Version | Hidden players |
|---|---|
| `truth` | true positions (reference) |
| `visible` | left out |
| `kalman` | the Kalman anchor (`imp_mean`) |
| `model` | the released model's mean |
Velocities come from each version's own positions 0.5 s earlier in the same window.

## 4. Analyses and measures
- **A. Pitch control** (`pitchvision.pitch_control`, 32 columns): share of grid cells where the controlling team
  (probability ≥ 0.5) is the same as with `truth`. All four versions.
- **B. Pass receiver** (spec 11, amendment 1 snapping; the window whose middle frame is nearest the pass, within 10
  frames): top-1 accuracy of the spec 11 `gnn`, trained with spec 11 settings on the other 4 IDSSE development
  matches. Versions `truth`, `kalman`, `model`. For `visible`: the share of passes whose true receiver is off-camera at
  that frame (they cannot be named without filling in).
- **C. Ghost team** (spec 10): mean error of the spec 10 `ghost` model (spec 10 settings, trained on the other 7
  development matches) when the other team's input is `truth`, `kalman` or `model`. Only 11 v 11 frames.

## 5. Reported (no pass rule)
For A: the **recovered share** = (model − visible) / (truth − visible), with truth = 100% agreement, and the same for
`kalman`; block bootstrap over 5-minute stretches (10,000 samples, seed 0). For B and C: each version's number and the
gap to `truth`, with the same bootstrap. A result where filling in recovers little is reported as such.

## 6. Output
`scripts/downstream.py`, `results/13-downstream.json`. README: the main results table of the project.
