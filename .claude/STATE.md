# State (2026-11-22)

The project question: broadcast tracking misses about half the players; how much analysis does that break, and how
much does filling them in bring back? README leads with it (spec 13 table).

## Branches
- main: pitch control + drag-and-react in the page (merged, pushed).
- research/ghost-team (NOT merged, not pushed; owner must ask): specs 10 (ghost team), 11 (pass receiver),
  13 (downstream study, recorded), README rewrite, site overlays (likely pass, expected defensive shape;
  `scripts/export_overlays.py` -> `site/demo/overlays.json`). `make check` passes; checked in headless Chromium
  at 1280 and 390 wide, light and dark, no console errors.

## Next
- Merge research/ghost-team to main and push when the owner says so; refresh `docs/site.png`.
- Spec 12 sampled futures (flow-matching head, Kaggle GPU): register first; fits the "what happens next" row.
- Pass model is overconfident (spec 11 P1 failed): a temperature fitted on a separate match would be a new spec.
- PFF World Cup 2022 data (owner fills the form) for a second locked test.

## Open items
- The spec 07 holdout is used up: never load it. New claims need new data.
- clips.json is 1.27 MB (observation context for the in-page Kalman).
- onnx_variants.py needs onnx, onnxruntime, onnxconverter-common (not project dependencies).
