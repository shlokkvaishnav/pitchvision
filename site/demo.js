// pitchvision demo: run the anchored model in the browser (onnxruntime-web) on 9 s clips.
/* global ort */
import { buildInputs, toTensors } from "./demo-core.js";

const CHI2_90 = 4.605170185988091;
const $ = (id) => document.getElementById(id);
let meta, session, clip, frame = 0, timer = null, pred = null;

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function runModel(c) {
  const { L, A, feeds } = buildInputs(meta, c);
  const t0 = performance.now();
  const out = await session.run(toTensors(ort, feeds));
  const ms = performance.now() - t0;
  const mean = out.mean.data, logStd = out.log_std.data;
  const py = c.python_mean.flat(2);
  let maxDiff = 0;
  for (let i = 0; i < py.length; i++) maxDiff = Math.max(maxDiff, Math.abs(py[i] - mean[i]) * 52.5);
  return { mean, logStd, ms, maxDiff, L, A };
}

function stats(c, p) {
  const errM = [], errK = [];
  let inside = 0, n = 0;
  for (let t = 0; t < p.L; t++) for (let a = 0; a < 22; a++) {
    if (c.visible[t][a]) continue;
    const i = (t * p.A + a) * 2, tx = c.truth_m[t][a][0], ty = c.truth_m[t][a][1];
    const mx = p.mean[i] * meta.scale_m[0], my = p.mean[i + 1] * meta.scale_m[1];
    const kx = c.anchor_mean[t][a][0] * meta.scale_m[0], ky = c.anchor_mean[t][a][1] * meta.scale_m[1];
    errM.push(Math.hypot(mx - tx, my - ty));
    errK.push(Math.hypot(kx - tx, ky - ty));
    const sx = Math.exp(p.logStd[i]) * meta.scale_m[0], sy = Math.exp(p.logStd[i + 1]) * meta.scale_m[1];
    inside += ((mx - tx) / sx) ** 2 + ((my - ty) / sy) ** 2 <= CHI2_90;
    n++;
  }
  const med = (v) => { const s = [...v].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
  return { n, model: med(errM), kalman: med(errK), cover: n ? inside / n : NaN };
}

function draw() {
  const cv = $("pitch"), g = cv.getContext("2d");
  const W = cv.width, H = cv.height, pad = 18;
  const sx = (W - 2 * pad) / 105, sy = (H - 2 * pad) / 68;
  const X = (x) => pad + (x + 52.5) * sx, Y = (y) => pad + (34 - y) * sy;
  g.fillStyle = css("--pitch"); g.fillRect(0, 0, W, H);
  g.strokeStyle = css("--lines"); g.lineWidth = 1.2;
  g.strokeRect(X(-52.5), Y(34), 105 * sx, 68 * sy);
  g.beginPath(); g.moveTo(X(0), Y(34)); g.lineTo(X(0), Y(-34)); g.stroke();
  g.beginPath(); g.arc(X(0), Y(0), 9.15 * sx, 0, 2 * Math.PI); g.stroke();
  for (const s of [-1, 1]) g.strokeRect(X(s > 0 ? 52.5 - 16.5 : -52.5), Y(20.16), 16.5 * sx, 40.32 * sy);

  const t = frame, A = pred.A, showTruth = $("truth").checked, showK = $("kalman").checked;
  for (let a = 0; a < 23; a++) {
    const [tx, ty] = clip.truth_m[t][a];
    const team = meta.agent_type[a];
    const color = team === 2 ? css("--ball") : team === 0 ? css("--home") : css("--away");
    const seen = clip.visible[t][a];
    if (seen) {
      g.fillStyle = color; g.beginPath(); g.arc(X(tx), Y(ty), team === 2 ? 3.5 : 5.5, 0, 2 * Math.PI); g.fill();
      continue;
    }
    const i = (t * A + a) * 2;
    const mx = pred.mean[i] * meta.scale_m[0], my = pred.mean[i + 1] * meta.scale_m[1];
    const rx = Math.sqrt(CHI2_90) * Math.exp(pred.logStd[i]) * meta.scale_m[0];
    const ry = Math.sqrt(CHI2_90) * Math.exp(pred.logStd[i + 1]) * meta.scale_m[1];
    g.globalAlpha = 0.18; g.fillStyle = color;
    g.beginPath(); g.ellipse(X(mx), Y(my), Math.max(rx * sx, 1), Math.max(ry * sy, 1), 0, 0, 2 * Math.PI); g.fill();
    g.globalAlpha = 1; g.strokeStyle = color; g.lineWidth = 2;
    g.beginPath(); g.moveTo(X(mx) - 4, Y(my) - 4); g.lineTo(X(mx) + 4, Y(my) + 4);
    g.moveTo(X(mx) - 4, Y(my) + 4); g.lineTo(X(mx) + 4, Y(my) - 4); g.stroke();
    if (showK) {
      const kx = clip.anchor_mean[t][a][0] * meta.scale_m[0], ky = clip.anchor_mean[t][a][1] * meta.scale_m[1];
      g.fillStyle = css("--muted"); g.beginPath(); g.arc(X(kx), Y(ky), 2.5, 0, 2 * Math.PI); g.fill();
    }
    if (showTruth) {
      g.strokeStyle = color; g.lineWidth = 1.5; g.setLineDash([2, 2]);
      g.beginPath(); g.arc(X(tx), Y(ty), 5.5, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
    }
  }
  $("time").textContent = `${(frame / meta.fps).toFixed(1)} s`;
  $("slider").value = frame;
}

async function load(k) {
  clip = meta.clips[k];
  $("status").textContent = "Running the model in your browser…";
  pred = await runModel(clip);
  const s = stats(clip, pred);
  $("status").textContent =
    `Model ran in ${pred.ms.toFixed(0)} ms. It agrees with the Python model to ${pred.maxDiff.toExponential(1)} m.`;
  $("sModel").textContent = `${s.model.toFixed(2)} m`;
  $("sKalman").textContent = `${s.kalman.toFixed(2)} m`;
  $("sCover").textContent = `${(100 * s.cover).toFixed(0)}%`;
  $("sN").textContent = s.n;
  $("slider").max = pred.L - 1;
  frame = 0; draw();
}

function play() {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "Play"; return; }
  $("play").textContent = "Pause";
  timer = setInterval(() => { frame = (frame + 1) % pred.L; draw(); }, 1000 / meta.fps);
}

async function main() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    [meta, session] = await Promise.all([
      fetch("demo/clips.json").then((r) => r.json()),
      ort.InferenceSession.create("demo/model.onnx", { executionProviders: ["wasm"] }),
    ]);
    $("weights").textContent = meta.weights;
    $("attrib").textContent = meta.attribution;
    meta.clips.forEach((c, k) => {
      const o = document.createElement("option");
      o.value = k; o.textContent = `Clip ${k + 1} (minute ${(c.start_s / 60).toFixed(1)})`;
      $("clip").appendChild(o);
    });
    $("clip").addEventListener("change", (e) => load(Number(e.target.value)));
    $("slider").addEventListener("input", (e) => { frame = Number(e.target.value); draw(); });
    $("play").addEventListener("click", play);
    for (const id of ["truth", "kalman"]) $(id).addEventListener("change", draw);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
    await load(0);
  } catch (err) {
    $("status").textContent = `The demo did not load: ${err.message}`;
  }
}
main();
