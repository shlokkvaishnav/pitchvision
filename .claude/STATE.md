# State (2026-11-24)

v2: a diffusion model that samples K = 20 paths per hidden player, tested by leave-one-match-out over all 10 matches.

## Branches
- main: v1 (transformer, spec 13 downstream study, site overlays). research/ghost-team is already in main.
- v2/diffusion-lomo (pushed, not merged): spec 14 (literature check: U2Diff family, Choi 2026 centroid voting as the
  published baseline), spec 15 (pre-registered protocol, amendment A1: v-prediction), `diffusion.py`, `centroid_vote.py`,
  `lomo.py`, `scripts/lomo_fold.py`, `scripts/lomo_summary.py`, `kaggle/lomo/` (10 fold kernels). `make check` passes.

## Status (2026-11-27)
- Fold 1 (idsse-J03WMX) is COMPLETE. Its output is in `results/15-lomo/idsse-J03WMX/` (summary and README); arrays, checkpoints and inputs are in `~/Documents/Archives/pitchvision-lomo/idsse-J03WMX/` (local only). Result: transformer median 2.60 m, diffusion 2.89 m, spline 3.03 m, Kalman 3.55 m; diffusion 90% coverage 0.81. Folds 2 to 5 were never launched and are deferred (owner decision: no more
  optional folds). No LOMO result is claimed anywhere in the repo or on the resume.
- Retrieval command used (the project's CLI at ~/.cache/kaggle-cli/bin/kaggle authenticates itself): `kaggle kernels output shlokkvaishnav/pitchvision-lomo-idsse-j03wmx -p <scratch dir>`, then the files were inspected and copied.

## Next (deferred)
1. Spec 15 amendment A2: 5 folds (held out: idsse-J03WMX, idsse-J03WOH, idsse-J03WPY, idsse-J03WR9, metrica-2), each
   using the other 9 matches. Headline = 5-fold leave-one-match-out CV over a 10-match pool. About 15 GPU hours.
2. Owner's order: pitchvision runs last on Kaggle. When told, push two folds at a time (repo root, clean tree):
   `sh kaggle/lomo/push_all.sh "spec 15 A2" idsse-J03WMX idsse-J03WOH` (uploads the code at HEAD), then when those
   finish `SKIP_DATASET=1 sh kaggle/lomo/push_all.sh x idsse-J03WPY idsse-J03WR9`, then
   `SKIP_DATASET=1 sh kaggle/lomo/push_all.sh x metrica-2`. Do not re-version the dataset between pushes.
3. Download each kernel's `results/` into `results/15-lomo/`, run `scripts/lomo_summary.py` -> `results/15-lomo.json`,
   append the result to spec 15; then the released all-10-match models (separate job), ONNX + browser timing, README/site.

## Running (2026-11-24)
- Dataset shlokkvaishnav/pitchvision-src-lomo versioned from 8f82af3 (do not re-version).
- Pushed: pitchvision-lomo-idsse-j03wmx (session 1). Push of idsse-J03WOH failed: 'Maximum batch GPU session count of 2' (owner's wa-cell-a was running).
- Not yet pushed: idsse-J03WOH, idsse-J03WPY, idsse-J03WR9, metrica-2. Push each with
  `~/.cache/kaggle-cli/bin/kaggle kernels push -p kaggle/lomo/<match>` (no dataset step); retry later on the batch-count error.
- When all 5 are COMPLETE: `kaggle kernels output shlokkvaishnav/pitchvision-lomo-<match-lowercase> -p results/15-lomo/<match>`,
  run scripts/lomo_summary.py, append to spec 15, make check, commit, push.

## Open items
- Spec 15 uses the spec 07 holdout matches on the owner's decision (2026-11-24); the CLAUDE.md rule now allows only that use.
- Centroid voting is online (no future frames); the other methods are offline smoothers. Say so next to its numbers.
- The spec 07 holdout is used up; v2 numbers are cross-validation, not a locked test.
