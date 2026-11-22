// Shared by the demo page and tests/js: build the model inputs for one clip.
// Hidden cells get zero: the model never sees the true position of a hidden player.
import { kalmanSmooth } from "./kalman.js";

// The observed positions of the clip's 23 tracks over its context, in metres. NaN where a track is not observed.
export function decodeContext(clip) {
  const { frames, tracks_mm: tracks } = clip.context, N = tracks.length;
  const obs = new Float64Array(frames * N * 2).fill(NaN);
  tracks.forEach((runs, n) => {
    for (const run of runs) {
      for (let j = 1; j < run.length; j += 2) {
        const i = ((run[0] + (j - 1) / 2) * N + n) * 2;
        obs[i] = run[j] / 1000; obs[i + 1] = run[j + 1] / 1000;
      }
    }
  });
  return { obs, T: frames, N };
}

// Kalman smoother over the context, cut to the window frames: position mean (m) and variance (m^2), (L, 23, 2).
// react = { frame, moved: Map(track -> [x, y]) }: forget every observation after window frame `frame` (players and
// ball), and observe each moved track at its new spot from `frame` to the end of the window.
export function smoothWindow(meta, clip, react = null) {
  const { obs, T, N } = decodeContext(clip);
  const L = clip.visible.length, before = clip.context.before;
  if (react) {
    for (let tc = before + react.frame + 1; tc < T; tc++) for (let n = 0; n < N; n++) { obs[(tc * N + n) * 2] = NaN; obs[(tc * N + n) * 2 + 1] = NaN; }
    for (const [n, [x, y]] of react.moved) {
      for (let tc = before + react.frame; tc < before + L; tc++) { obs[(tc * N + n) * 2] = x; obs[(tc * N + n) * 2 + 1] = y; }
    }
  }
  const k = meta.kalman;
  const { mean, variance } = kalmanSmooth(obs, T, N, meta.fps, k.accel_std, k.obs_std);
  const from = before * N * 2, to = from + L * N * 2;
  return { mean: mean.subarray(from, to), variance: variance.subarray(from, to) };
}

// The model's anchor from the smoother, as pitchvision.anchor._normalise: mean / scale, clipped; log(std / scale).
export function anchorFromKalman(meta, sm) {
  const { min_std_m: minStd, clip: lim } = meta.kalman;
  const n = sm.mean.length, mean = new Float32Array(n), logStd = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = meta.scale_m[i % 2];
    mean[i] = Math.min(lim, Math.max(-lim, sm.mean[i] / s));
    logStd[i] = Math.log(Math.sqrt(Math.max(sm.variance[i], minStd * minStd)) / s);
  }
  return { mean, logStd };
}

// react (optional): see smoothWindow. Frames after react.frame are hidden for every track except the moved ones.
export function buildInputs(meta, clip, react = null) {
  const L = clip.truth_m.length, A = clip.truth_m[0].length;
  const pos = new Float32Array(L * A * 2), obs = new Float32Array(L * A);
  for (let t = 0; t < L; t++) for (let a = 0; a < A; a++) {
    const k = t * A + a, moved = react && t >= react.frame && react.moved.has(a);
    let seen = clip.visible[t][a];
    if (react && t > react.frame) seen = 0;
    let xy = clip.truth_m[t][a];
    if (moved) { seen = 1; xy = react.moved.get(a); }
    obs[k] = seen;
    if (seen) {
      pos[2 * k] = xy[0] / meta.scale_m[0];
      pos[2 * k + 1] = xy[1] / meta.scale_m[1];
    }
  }
  const anchor = anchorFromKalman(meta, smoothWindow(meta, clip, react));
  return {
    L, A,
    feeds: {
      pos: { type: "float32", data: pos, dims: [1, L, A, 2] },
      observed: { type: "float32", data: obs, dims: [1, L, A] },
      agent_type: { type: "int64", data: BigInt64Array.from(meta.agent_type.map(BigInt)), dims: [A] },
      anchor_mean: { type: "float32", data: anchor.mean, dims: [1, L, A, 2] },
      anchor_log_std: { type: "float32", data: anchor.logStd, dims: [1, L, A, 2] },
    },
  };
}

export function toTensors(ort, feeds) {
  const out = {};
  for (const [k, v] of Object.entries(feeds)) out[k] = new ort.Tensor(v.type, v.data, v.dims);
  return out;
}
