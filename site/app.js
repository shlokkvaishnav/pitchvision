/* global ort */
import { buildInputs, toTensors } from "./demo-core.js";
import { controlMap, velocities } from "./pitchcontrol.js";

const CHI2_90 = 4.605170185988091;
const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
let meta, session, clip, pred, frame = 0, playing = !still, last = 0;
let est = null, moved = new Map(), dragging = -1, ctrlCv = null; // est: per-frame positions the page knows (seen or estimated)

// ---------- model ----------
async function run(c) {
  const { L, A, feeds } = buildInputs(meta, c);
  const out = await session.run(toTensors(ort, feeds));
  return { mean: out.mean.data, logStd: out.log_std.data, L, A };
}

// ---------- pitch ----------
const cv = $("pitch"), g = cv.getContext("2d");
function sizeCanvas() {
  const r = cv.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.width * (68 / 105) * dpr);
}


// ---------- pitch control ----------
// Positions the page uses at frame f: the camera position if seen, else the model's estimate; dragged players moved.
function estSeries() {
  const A = pred.A;
  return clip.truth_m.map((fr, t) => fr.slice(0, 22).map((xy, a) => {
    if (clip.visible[t][a]) return xy;
    const i = (t * A + a) * 2;
    return [pred.mean[i] * meta.scale_m[0], pred.mean[i + 1] * meta.scale_m[1]];
  }));
}
function positionsAt(f) { return est[f].map((xy, a) => (f === frame && moved.has(a) ? moved.get(a) : xy)); }
function controlNow(series, pos) {
  const vel = velocities(series, frame, meta.fps).map((v, a) => (moved.has(a) ? [0, 0] : v));
  return controlMap(pos, vel, meta.agent_type.slice(0, 22), clip.truth_m[frame][22], 48);
}
function drawControl(X, Y, s) {
  const m = controlNow(est, positionsAt(frame));
  // one pixel per cell in a small image, scaled up with smoothing (no seams between cells)
  const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const home = rgb(css("--accent")), away = rgb(css("--cool"));
  if (!ctrlCv || ctrlCv.width !== m.nx) { ctrlCv = document.createElement("canvas"); ctrlCv.width = m.nx; ctrlCv.height = m.ny; }
  const cg = ctrlCv.getContext("2d"), img = cg.createImageData(m.nx, m.ny);
  for (let j = 0; j < m.ny; j++) for (let i = 0; i < m.nx; i++) {
    const p = m.home[j * m.nx + i], c = p >= 0.5 ? home : away, o = ((m.ny - 1 - j) * m.nx + i) * 4; // row 0 = top (y = +34)
    img.data.set([c[0], c[1], c[2], Math.round(Math.min(1, Math.abs(p - 0.5) * 2) * 0.42 * 255)], o);
  }
  cg.putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(ctrlCv, X(-52.5), Y(34), 105 * s, 68 * s);
  g.globalAlpha = 1;
  let text;
  if ($("truth").checked && !moved.size) {
    const t = controlNow(clip.truth_m.map((fr) => fr.slice(0, 22)), clip.truth_m[frame].slice(0, 22));
    let same = 0;
    for (let i = 0; i < m.home.length; i++) same += (m.home[i] >= 0.5) === (t.home[i] >= 0.5);
    text = `Control matches the true positions on <b>${Math.round((100 * same) / m.home.length)}%</b>`;
  } else {
    const away = 1 - m.home.reduce((a, p) => a + p, 0) / m.home.length;
    text = `Blue controls <b>${Math.round(away * 100)}%</b>`;
  }
  $("ctrl").innerHTML = text; $("ctrl").hidden = false;
}

function draw() {
  if (!pred) return;
  const W = cv.width, H = cv.height, s = W / 105;
  const X = (x) => (x + 52.5) * s, Y = (y) => (34 - y) * s;
  const k = W / 1000; // line and marker scale
  // grass stripes
  for (let i = 0; i < 14; i++) { g.fillStyle = i % 2 ? css("--pitch-b") : css("--pitch-a"); g.fillRect((i * W) / 14, 0, W / 14 + 1, H); }
  // off-camera shade: everything except the camera footprint
  const cam = clip.camera_m[frame];
  g.save(); g.beginPath(); g.rect(0, 0, W, H);
  g.moveTo(X(cam[0][0]), Y(cam[0][1])); for (let i = 3; i >= 1; i--) g.lineTo(X(cam[i][0]), Y(cam[i][1])); g.closePath();
  g.fillStyle = css("--shade"); g.fill("evenodd"); g.restore();
  // markings
  g.strokeStyle = css("--pitch-line"); g.lineWidth = 2.2 * k;
  g.strokeRect(X(-52.5) + k, Y(34) + k, 105 * s - 2 * k, 68 * s - 2 * k);
  g.beginPath(); g.moveTo(X(0), Y(34)); g.lineTo(X(0), Y(-34)); g.stroke();
  g.beginPath(); g.arc(X(0), Y(0), 9.15 * s, 0, 2 * Math.PI); g.stroke();
  for (const e of [-1, 1]) {
    const x0 = e < 0 ? -52.5 : 52.5 - 16.5;
    g.strokeRect(X(x0), Y(20.16), 16.5 * s, 40.32 * s);
    g.strokeRect(X(e < 0 ? -52.5 : 52.5 - 5.5), Y(9.16), 5.5 * s, 18.32 * s);
  }
  if ($("control").checked) drawControl(X, Y, s); else $("ctrl").hidden = true;
  // camera outline
  g.beginPath(); cam.forEach(([x, y], i) => (i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)))); g.closePath();
  g.strokeStyle = css("--pitch-line"); g.setLineDash([8 * k, 6 * k]); g.lineWidth = 2 * k; g.stroke(); g.setLineDash([]);

  const reveal = $("truth").checked, A = pred.A;
  let hiddenCount = 0;
  const team = (a) => (meta.agent_type[a] === 0 ? css("--accent") : css("--cool"));
  // hidden players first, under the visible ones
  for (let a = 0; a < 22; a++) {
    if (clip.visible[frame][a]) continue;
    hiddenCount++;
    const i = (frame * A + a) * 2;
    const [mx, my] = moved.has(a) ? moved.get(a) : [pred.mean[i] * meta.scale_m[0], pred.mean[i + 1] * meta.scale_m[1]];
    const rx = Math.sqrt(CHI2_90) * (meta.std_scale ?? 1) * Math.exp(pred.logStd[i]) * meta.scale_m[0];
    const ry = Math.sqrt(CHI2_90) * (meta.std_scale ?? 1) * Math.exp(pred.logStd[i + 1]) * meta.scale_m[1];
    const c = team(a);
    g.globalAlpha = 0.16; g.fillStyle = c;
    g.beginPath(); g.ellipse(X(mx), Y(my), Math.max(rx * s, 4 * k), Math.max(ry * s, 4 * k), 0, 0, 2 * Math.PI); g.fill();
    g.globalAlpha = 1; g.strokeStyle = c; g.lineWidth = 3 * k;
    g.beginPath(); g.arc(X(mx), Y(my), 8 * k, 0, 2 * Math.PI); g.stroke();
    if (reveal) {
      const [tx, ty] = clip.truth_m[frame][a];
      g.strokeStyle = css("--ink"); g.globalAlpha = 0.55; g.lineWidth = 1.5 * k;
      g.beginPath(); g.moveTo(X(mx), Y(my)); g.lineTo(X(tx), Y(ty)); g.stroke(); g.globalAlpha = 1;
      g.fillStyle = css("--ink"); g.beginPath(); g.arc(X(tx), Y(ty), 4 * k, 0, 2 * Math.PI); g.fill();
    }
  }
  for (let a = 0; a < 23; a++) {
    if (a < 22 && !clip.visible[frame][a]) continue;
    const [x, y] = a < 22 && moved.has(a) ? moved.get(a) : clip.truth_m[frame][a];
    if (a === 22) { g.fillStyle = css("--ink"); g.beginPath(); g.arc(X(x), Y(y), 5 * k, 0, 2 * Math.PI); g.fill(); continue; }
    g.fillStyle = team(a); g.strokeStyle = css("--card"); g.lineWidth = 2.5 * k;
    g.beginPath(); g.arc(X(x), Y(y), 9 * k, 0, 2 * Math.PI); g.fill(); g.stroke();
  }
  $("nHidden").textContent = hiddenCount;
  $("clock").textContent = `${(frame / meta.fps).toFixed(1)} s`;
}

function tick(t) {
  if (playing && pred && t - last >= 1000 / meta.fps) { frame = (frame + 1) % pred.L; last = t; moved.clear(); draw(); }
  requestAnimationFrame(tick);
}

async function pick(k) {
  clip = meta.clips[k];
  for (const b of $("clips").children) b.setAttribute("aria-pressed", String(Number(b.dataset.k) === k));
  pred = await run(clip);
  est = estSeries(); moved.clear();
  frame = 0; draw();
}

// ---------- how it works ----------
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const [a, v] of Object.entries(attrs)) e.setAttribute(a, v); parent.appendChild(e); return e; }
function label(parent, x, y, s, attrs = {}) { const t = el("text", { x, y, ...attrs }, parent); t.textContent = s; return t; }

function drawFlow() {
  const svg = $("flow"); svg.replaceChildren();
  const box = svg.parentElement.getBoundingClientRect();
  const W = Math.max(300, Math.round(box.width - 2 * parseFloat(getComputedStyle(svg.parentElement).paddingLeft)));
  const narrow = W < 560, n = 4;
  const steps = [
    ["Broadcast", "only the players on camera"],
    ["Kalman track", "a first guess from motion"],
    ["Transformer", "corrects it with all 22 players"],
    ["Full pitch", "every player, with a region"],
  ];
  const cw = narrow ? W : W / n, ch = narrow ? 92 : 170, H = narrow ? ch * n : ch;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const ink = css("--ink"), mut = css("--muted"), acc = css("--accent"), cool = css("--cool"), chip = css("--chip");
  steps.forEach(([t, s], i) => {
    const cx = narrow ? 52 : cw * i + cw / 2, cy = narrow ? ch * i + 40 : 58;
    const gI = el("g", { transform: `translate(${cx} ${cy})` }, svg);
    el("rect", { x: -40, y: -30, width: 80, height: 60, rx: 14, fill: chip }, gI);
    if (i === 0) { el("rect", { x: -26, y: -16, width: 52, height: 32, rx: 4, fill: "none", stroke: ink, "stroke-width": 2 }, gI); el("circle", { cx: -8, cy: 0, r: 4, fill: acc }, gI); el("circle", { cx: 9, cy: -5, r: 4, fill: cool }, gI); }
    if (i === 1) el("path", { d: "M-28 12 C-14 12 -14 -12 0 -12 S 14 8 28 -6", fill: "none", stroke: ink, "stroke-width": 2.5, "stroke-dasharray": "4 4" }, gI);
    if (i === 2) for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) el("rect", { x: -26 + c * 14, y: -18 + r * 13, width: 10, height: 9, rx: 2, fill: (r + c) % 3 ? ink : acc, opacity: (r + c) % 3 ? 0.35 : 1 }, gI);
    if (i === 3) { el("ellipse", { cx: -8, cy: 2, rx: 16, ry: 11, fill: acc, opacity: 0.2 }, gI); el("circle", { cx: -8, cy: 2, r: 5, fill: "none", stroke: acc, "stroke-width": 2.5 }, gI); el("circle", { cx: 14, cy: -8, r: 5, fill: cool }, gI); }
    if (narrow) {
      label(svg, 110, cy - 4, t, { "font-size": 15, "font-weight": 600, fill: ink, "font-family": "var(--serif)" });
      label(svg, 110, cy + 16, s, { "font-size": 13, fill: mut });
      if (i < n - 1) el("path", { d: `M${cx} ${cy + 34} v${ch - 68}`, stroke: mut, "stroke-width": 1.5, "marker-end": "url(#ah)" }, svg);
    } else {
      label(svg, cx, 120, t, { "font-size": 16, "font-weight": 600, fill: ink, "text-anchor": "middle", "font-family": "var(--serif)" });
      label(svg, cx, 142, s, { "font-size": 13, fill: mut, "text-anchor": "middle" });
      if (i < n - 1) el("path", { d: `M${cx + 50} ${cy} H${cx + cw - 54}`, stroke: mut, "stroke-width": 1.5, "marker-end": "url(#ah)" }, svg);
    }
  });
  const defs = el("defs", {}, svg), m = el("marker", { id: "ah", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
  el("path", { d: "M0 0 L10 5 L0 10 z", fill: mut }, m);
}

// ---------- how much the camera hides ----------
function drawGaps() {
  $("share").textContent = `${Math.round(meta.hidden_share * 100)}%`;
  const b = meta.hidden_gap_bins;
  const parts = [
    ["under 3 s", b["<1s"] + b["1-3s"], css("--cool-soft")],
    ["3 to 10 s", b["3-6s"] + b["6-9.6s"], css("--cool")],
    ["over 10 s", b[">=9.6s"], css("--accent")],
  ];
  const svg = $("gaps"); svg.replaceChildren();
  const W = Math.max(300, Math.round(svg.parentElement.getBoundingClientRect().width - 2 * parseFloat(getComputedStyle(svg.parentElement).paddingLeft)));
  const narrow = W < 420;
  svg.setAttribute("viewBox", `0 0 ${W} ${narrow ? 150 : 104}`);
  label(svg, 0, 16, "How long a hidden player stays hidden", { "font-size": 13, fill: css("--muted") });
  let x = 0;
  parts.forEach(([, f, c], i) => {
    const w = f * W;
    el("rect", { x: x + (i ? 2 : 0), y: 28, width: Math.max(w - (i ? 2 : 0), 2), height: 34, rx: 6, fill: c }, svg);
    if (w > 70) label(svg, x + 12, 51, `${Math.round(f * 100)}%`, { "font-size": 15, "font-weight": 700, fill: i ? css("--paper") : css("--ink") });
    x += w;
  });
  // key row: one entry per part, evenly spaced (stacked on a phone)
  parts.forEach(([name, f, c], i) => {
    const kx = narrow ? 0 : (i * W) / parts.length, ky = narrow ? 88 + i * 22 : 90;
    el("rect", { x: kx, y: ky - 10, width: 12, height: 12, rx: 3, fill: c }, svg);
    label(svg, kx + 18, ky, `${name}`, { "font-size": 13, fill: css("--ink") });
    label(svg, kx + 18 + name.length * 6.6 + 8, ky, `${Math.round(f * 100)}%`, { "font-size": 13, "font-weight": 700, fill: css("--muted") });
  });
}


// ---------- what if: drag a player ----------
function pitchXY(ev) {
  const r = cv.getBoundingClientRect();
  return [((ev.clientX - r.left) / r.width) * 105 - 52.5, 34 - ((ev.clientY - r.top) / r.height) * 68];
}
function setPlaying(v) { playing = v; $("play").textContent = playing ? "Pause" : "Play"; cv.classList.toggle("drag", !playing); }
cv.addEventListener("pointerdown", (ev) => {
  if (!pred) return;
  const [x, y] = pitchXY(ev), pos = positionsAt(frame);
  let best = -1, bd = 3.5;
  pos.forEach(([px, py], a) => { const d = Math.hypot(px - x, py - y); if (d < bd) { bd = d; best = a; } });
  if (best < 0) return;
  setPlaying(false); dragging = best; cv.setPointerCapture(ev.pointerId); ev.preventDefault();
});
cv.addEventListener("pointermove", (ev) => {
  if (dragging < 0) return;
  const [x, y] = pitchXY(ev);
  moved.set(dragging, [Math.max(-52.5, Math.min(52.5, x)), Math.max(-34, Math.min(34, y))]); draw();
});
const endDrag = () => { dragging = -1; };
cv.addEventListener("pointerup", endDrag); cv.addEventListener("pointercancel", endDrag);

// ---------- start ----------
async function main() {
  sizeCanvas();
  try {
    // same version as the ort.min.js tag in index.html (and package.json, for the Node tests)
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
    [meta, session] = await Promise.all([
      fetch("demo/clips.json").then((r) => r.json()),
      ort.InferenceSession.create("demo/model.onnx", { executionProviders: ["wasm"] }),
    ]);
    meta.clips.forEach((c, k) => {
      const b = document.createElement("button");
      b.textContent = `Clip ${k + 1}`; b.dataset.k = k; b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => pick(k));
      $("clips").appendChild(b);
    });
    setPlaying(playing);
    $("play").addEventListener("click", () => { moved.clear(); setPlaying(!playing); });
    $("control").addEventListener("change", draw);
    $("reset").addEventListener("click", () => { moved.clear(); draw(); });
    $("truth").addEventListener("change", draw);
    drawFlow(); drawGaps();
    await pick(0);
    $("status").textContent = "";
    requestAnimationFrame(tick);
  } catch (err) {
    $("status").textContent = `The demo could not load: ${err.message}`;
  }
}
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { sizeCanvas(); draw(); if (meta) { drawFlow(); drawGaps(); } }, 120); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { draw(); if (meta) { drawFlow(); drawGaps(); } });
main();
