# State (2026-11-24)

v2: a diffusion model that samples K = 20 paths per hidden player, tested by leave-one-match-out over all 10 matches.

## Branches
- main: v1 (transformer, spec 13 downstream study, site overlays). research/ghost-team is already in main.
- v2/diffusion-lomo (pushed, not merged): spec 14 (literature check: U2Diff family, Choi 2026 centroid voting as the
  published baseline), spec 15 (pre-registered protocol, amendment A1: v-prediction), `diffusion.py`, `centroid_vote.py`,
  `lomo.py`, `scripts/lomo_fold.py`, `scripts/lomo_summary.py`, `kaggle/lomo/` (10 fold kernels). `make check` passes.

## Next
1. Push the fold kernels. On 2026-11-24 every push failed: "Maximum batch GPU session count of 2 reached" (two
   ghostdesk kernels were queued). The dataset `pitchvision-src-lomo` holds commit 5cfdb08. When a slot is free:
   `SKIP_DATASET=1 sh kaggle/lomo/push_all.sh x idsse-J03WMX idsse-J03WN1` (2 at a time; do not re-version the dataset).
   About 3 GPU hours per fold, about 30 in all: more than one week of Kaggle quota.
2. Download each kernel's `results/` into `results/15-lomo/`, then `scripts/lomo_summary.py` -> `results/15-lomo.json`.
3. Append the result to spec 15; then the released models (spec 15 section 8), ONNX diffusion + browser timing, README/site.

## Open items
- Spec 15 uses the spec 07 holdout matches on the owner's decision; CLAUDE.md still says never load them (owner to update).
- Centroid voting is online (no future frames); the other methods are offline smoothers. Say so next to its numbers.
- The spec 07 holdout is used up; v2 numbers are cross-validation, not a locked test.
