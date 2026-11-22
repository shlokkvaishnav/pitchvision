// The page's pitch control (site/pitchcontrol.js) must give the Python reference (pitchvision.pitch_control).
// tests/js/pitch-control-ref.json is written by scripts/export_pitch_control_ref.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { controlMap, velocities } from "../../site/pitchcontrol.js";

const meta = JSON.parse(readFileSync(new URL("../../site/demo/clips.json", import.meta.url)));
const ref = JSON.parse(readFileSync(new URL("./pitch-control-ref.json", import.meta.url)));

test("JS pitch control equals Python on demo frames", () => {
  const team = meta.agent_type.slice(0, 22);
  for (const c of ref.cases) {
    const truth = meta.clips[c.clip].truth_m;
    const series = truth.map((fr) => fr.slice(0, 22));
    const m = controlMap(series[c.frame], velocities(series, c.frame, meta.fps), team, truth[c.frame][22], ref.nx);
    let d = 0;
    for (let i = 0; i < m.home.length; i++) d = Math.max(d, Math.abs(m.home[i] - c.home[i]));
    console.log(`clip ${c.clip} frame ${c.frame}: max diff ${d.toExponential(2)}`);
    assert.ok(d < 1e-6, `max diff ${d}`);
  }
});
