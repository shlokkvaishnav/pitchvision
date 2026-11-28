# State (2026-11-28)

v2: a diffusion model that samples K = 20 paths per hidden player, tested by leave-one-match-out (spec 15, amendment A2: 5 folds).

## Branches
- main: v1 (transformer, spec 13, site overlays).
- v2/diffusion-lomo (pushed, not merged): specs 14 and 15, diffusion code, `kaggle/lomo/` fold kernels, fold results. `make check` passes.

## Spec 15 folds (dataset pitchvision-src-lomo from 8f82af3; do not re-version)
- Fold 1 idsse-J03WMX COMPLETE: transformer 2.60 m, diffusion 2.89, spline 3.03, Kalman 3.55 (median).
- Fold 2 idsse-J03WOH COMPLETE: transformer 2.76, diffusion 2.98, spline 3.36, Kalman 3.88. Diffusion 90% coverage 0.82.
- Running on Kaggle: idsse-J03WPY, idsse-J03WR9. Not pushed: metrica-2
  (`SKIP_DATASET=1 sh kaggle/lomo/push_all.sh "spec 15 A2" metrica-2` when a slot frees).
- Per fold: `kaggle kernels output shlokkvaishnav/pitchvision-lomo-<match-lower> -p <scratch>`; json to
  `results/15-lomo/<match>/` with a README; npz and .pt to `data/15-lomo/<match>/` (gitignored); delete raw skillcorner/.
- All 5 done: `PYTHONPATH=src python scripts/lomo_summary.py`, append the result to spec 15, make check, commit, push.

## Next: spec 16 (owner approved 2026-11-28, after the folds)
- Idea: residual diffusion on the transformer. The transformer gives the point estimate; diffusion samples only the
  residual around it. Accuracy is held at the transformer's by design; the claim is better energy score and coverage.
- Step 1: literature check (residual or mean-plus-diffusion, e.g. CorrDiff in weather, CSDI-style imputers, multi-agent
  trajectory imputation). Cite only papers opened. Step 2: register spec 16 before any run.
- Data: the 10 matches are used by spec 15, so spec 16 on them is exploratory; a confirmatory claim needs new matches.

## Open items
- Centroid voting is online (no future frames); the others are offline smoothers. Say so next to its numbers.
- Why diffusion trails (for the spec 15 write-up): no fitted spread scale (transformer has one, about 1.10), eta = 0
  chosen, coverage falls to 0.79 at the longest gaps.
