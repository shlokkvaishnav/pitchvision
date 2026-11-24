# pitchvision: notes for the agent

Fills in the football players a TV broadcast does not show, from tracking data (x, y), not video:
a Kalman smoother gives a first estimate, an axial transformer corrects it using all 22 players and
the ball, and a fitted scale makes the 90% regions honest. Both steps run in the browser (plain-JS
Kalman, ONNX via onnxruntime-web). Live: https://shlokkvaishnav.github.io/pitchvision/.

## Start here
1. The SessionStart hook prints the branch and the next step from `.claude/STATE.md`.
2. `README.md` (results), `docs/method.md` (how it was tested), `docs/specs/NN-*.md` (each experiment,
   design first, result appended). Code in `src/pitchvision/`, scripts in `scripts/`, site in `site/`.

## Hard rules
- MUST register a spec in `docs/specs/` and commit it before any new evaluation; never edit a
  registered rule after its result exists.
- The locked holdout (two IDSSE matches, spec 07) is used up. The only new use allowed is spec 15
  (leave-one-match-out over all 10 matches, owner's decision 2026-11-24): its numbers are
  cross-validation, not a fresh locked test, and must be labelled so. Any other new use of those two matches
  is exploratory and labelled so. New confirmatory claims need new data.
- The browser Kalman (`site/kalman.js`) must match the Python one: keep `tests/js` parity tests green.
- CDN scripts are pinned to an exact version with an SRI hash.
- Git: work on a branch; merge and push only when the owner asks; `make check` before any push.
  No co-author or AI attribution lines. Plain English; README for engineers, no process notes.
- Never Read tracking data, `site/demo/clips.json` or model files whole; use short scripts.

## Commands (repo root)
- Setup: `make install`. Done check: `make check` (ruff, mypy, pytest, Node parity tests).
- Rebuild the demo data: `scripts/export_demo.py`; reproduce results: `make reproduce`.

## Definition of done
`make check` passes; numbers in README/method come from `results/` files written by committed scripts;
the page renders (desktop and phone width). One task at a time.

## End of session
Rewrite `.claude/STATE.md` (about 20 lines): branch, next step, open items. Replace it.
