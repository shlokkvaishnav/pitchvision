// The JavaScript Kalman smoother (site/kalman.js) must give the Python smoother (pitchvision.baselines.kalman).
// tests/js/kalman-ref.json: Python output on the same decoded context, written by scripts/export_demo.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { smoothWindow } from "../../site/demo-core.js";

const meta = JSON.parse(readFileSync(new URL("../../site/demo/clips.json", import.meta.url)));
const ref = JSON.parse(readFileSync(new URL("./kalman-ref.json", import.meta.url)));

// Both sides compute in float64; the reference is rounded to 1e-9 m and 1e-12 m^2.
const TOL_MEAN_M = 1e-6, TOL_VAR_M2 = 1e-6;

test("JS Kalman smoother equals the Python smoother on all demo clips", () => {
  assert.equal(ref.clips.length, meta.clips.length);
  meta.clips.forEach((clip, c) => {
    const sm = smoothWindow(meta, clip);
    const pm = ref.clips[c].mean_m.flat(2), pv = ref.clips[c].var_m2.flat(2);
    assert.equal(sm.mean.length, pm.length);
    let dm = 0, dv = 0;
    for (let i = 0; i < pm.length; i++) {
      dm = Math.max(dm, Math.abs(sm.mean[i] - pm[i]));
      dv = Math.max(dv, Math.abs(sm.variance[i] - pv[i]));
    }
    console.log(`clip ${c + 1}: max |mean diff| ${dm.toExponential(2)} m, max |var diff| ${dv.toExponential(2)} m^2`);
    assert.ok(dm < TOL_MEAN_M && dv < TOL_VAR_M2, `clip ${c + 1}: mean diff ${dm} m, var diff ${dv} m^2`);
  });
});
