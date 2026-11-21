// Time onnxruntime-web (wasm, Node) on the demo clips for each ONNX file given. Prints a JSON list: median ms per clip.
// Used by scripts/onnx_variants.py. Usage: node scripts/bench_web.mjs model-a.onnx model-b.onnx ...
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { buildInputs, toTensors } from "../site/demo-core.js";

const meta = JSON.parse(readFileSync(new URL("../site/demo/clips.json", import.meta.url)));
const inputs = meta.clips.map((c) => toTensors(ort, buildInputs(meta, c).feeds));
const out = [];
for (const path of process.argv.slice(2)) {
  const session = await ort.InferenceSession.create(readFileSync(path), { executionProviders: ["wasm"] });
  for (const f of inputs) await session.run(f); // warm-up
  const times = [];
  for (let r = 0; r < 20; r++) for (const f of inputs) {
    const t0 = performance.now();
    await session.run(f);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  out.push(times[Math.floor(times.length / 2)]);
}
console.log(JSON.stringify(out));
