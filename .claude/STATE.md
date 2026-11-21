# State (2026-11-21)

## Branches
- main: Kalman smoother runs in the page (JS port, parity tests), onnxruntime-web 1.31.1 with SRI,
  spec 09 camera stress test and fp16/int8 note (exploratory), README notes the locked result is
  mostly one match (J03WOH, 2.76M of 2.93M positions).

## Next
- New data is needed for any new claim (the holdout is used up), e.g. more IDSSE or other open
  tracking matches; register a spec first.

## Open items
- clips.json grew to 1.27 MB (observation context for the in-page Kalman).
- onnx_variants.py needs onnx, onnxruntime, onnxconverter-common (not project dependencies).
- Upstream idea: kloppy's per-point coordinate conversion is slow (src/pitchvision/tracking.py
  works around it); check current kloppy before proposing a vectorised PR.
