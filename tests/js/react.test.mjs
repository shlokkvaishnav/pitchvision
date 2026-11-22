// "Players react to a drag": after the paused frame the model must see only the dragged players, never the real future.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { buildInputs, toTensors } from "../../site/demo-core.js";

const meta = JSON.parse(readFileSync(new URL("../../site/demo/clips.json", import.meta.url)));
const modelBytes = readFileSync(new URL("../../site/demo/model.onnx", import.meta.url));
const F = 30, MOVED = 3, SPOT = [10, -5];

// Same clip with every position after window frame F shifted by 7 m (truth and context): a different real future.
function otherFuture(clip) {
  const c = structuredClone(clip), before = c.context.before;
  c.truth_m = c.truth_m.map((fr, t) => (t > F ? fr.map(([x, y]) => [x + 7, y - 7]) : fr));
  c.context.tracks_mm = c.context.tracks_mm.map((runs) => runs.map((run) => run.map((v, j) => {
    if (j === 0) return v;
    const frame = run[0] + Math.floor((j - 1) / 2);
    return frame > before + F ? v + ((j - 1) % 2 === 0 ? 7000 : -7000) : v;
  })));
  return c;
}

test("react mode hides the real future and holds the dragged player", async () => {
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  const clip = meta.clips[0], react = { frame: F, moved: new Map([[MOVED, SPOT]]) };
  const a = buildInputs(meta, clip, react), b = buildInputs(meta, otherFuture(clip), react);
  const A = a.A, obs = a.feeds.observed.data;
  for (let t = F + 1; t < a.L; t++) for (let k = 0; k < A; k++) assert.equal(obs[t * A + k], k === MOVED ? 1 : 0);
  const ra = await session.run(toTensors(ort, a.feeds)), rb = await session.run(toTensors(ort, b.feeds));
  let d = 0;
  for (let i = 0; i < ra.mean.data.length; i++) d = Math.max(d, Math.abs(ra.mean.data[i] - rb.mean.data[i]));
  assert.equal(d, 0, `output depends on the real future (max diff ${d})`);
  const i = ((a.L - 1) * A + MOVED) * 2;
  const end = [ra.mean.data[i] * meta.scale_m[0], ra.mean.data[i + 1] * meta.scale_m[1]];
  console.log("dragged player at the last frame:", end.map((v) => v.toFixed(2)));
  assert.ok(Math.hypot(end[0] - SPOT[0], end[1] - SPOT[1]) < 1.0);
});
