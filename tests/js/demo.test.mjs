// The whole page pipeline (JS Kalman smoother, then the ONNX model in onnxruntime-web) must give the Python predictions
// (Python smoother on the same context, then PyTorch).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { buildInputs, toTensors } from "../../site/demo-core.js";

const meta = JSON.parse(readFileSync(new URL("../../site/demo/clips.json", import.meta.url)));
const modelBytes = readFileSync(new URL("../../site/demo/model.onnx", import.meta.url));

test("JS Kalman + onnxruntime-web equals Python Kalman + PyTorch on all demo clips", async () => {
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  for (const clip of meta.clips) {
    const { feeds } = buildInputs(meta, clip);
    const out = await session.run(toTensors(ort, feeds));
    const py = clip.python_mean.flat(2), pls = clip.python_log_std.flat(2);
    let dm = 0, dl = 0;
    for (let i = 0; i < py.length; i++) {
      dm = Math.max(dm, Math.abs(out.mean.data[i] - py[i]));
      dl = Math.max(dl, Math.abs(out.log_std.data[i] - pls[i]));
    }
    console.log(`mean diff ${dm.toExponential(2)}, log_std diff ${dl.toExponential(2)} (normalised)`);
    // 1e-4 in normalised units is about 5 mm on the pitch; python_mean and python_log_std are rounded to 5 decimals
    assert.ok(dm < 1e-4 && dl < 1e-4, `mean diff ${dm}, log_std diff ${dl}`);
  }
});
