// Shared by the demo page and tests/js/demo.test.mjs: build the model inputs for one clip.
// Hidden cells get zero: the model never sees the true position of a hidden player.
export function buildInputs(meta, clip) {
  const L = clip.truth_m.length, A = clip.truth_m[0].length;
  const pos = new Float32Array(L * A * 2), obs = new Float32Array(L * A);
  for (let t = 0; t < L; t++) for (let a = 0; a < A; a++) {
    const k = t * A + a, seen = clip.visible[t][a];
    obs[k] = seen;
    if (seen) {
      pos[2 * k] = clip.truth_m[t][a][0] / meta.scale_m[0];
      pos[2 * k + 1] = clip.truth_m[t][a][1] / meta.scale_m[1];
    }
  }
  return {
    L, A,
    feeds: {
      pos: { type: "float32", data: pos, dims: [1, L, A, 2] },
      observed: { type: "float32", data: obs, dims: [1, L, A] },
      agent_type: { type: "int64", data: BigInt64Array.from(meta.agent_type.map(BigInt)), dims: [A] },
      anchor_mean: { type: "float32", data: Float32Array.from(clip.anchor_mean.flat(2)), dims: [1, L, A, 2] },
      anchor_log_std: { type: "float32", data: Float32Array.from(clip.anchor_log_std.flat(2)), dims: [1, L, A, 2] },
    },
  };
}

export function toTensors(ort, feeds) {
  const out = {};
  for (const [k, v] of Object.entries(feeds)) out[k] = new ort.Tensor(v.type, v.data, v.dims);
  return out;
}
