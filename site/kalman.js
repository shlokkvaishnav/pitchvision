// Kalman filter and Rauch-Tung-Striebel smoother with a constant-velocity model, in plain JavaScript.
// A port of pitchvision.baselines.kalman (Python): same initial state, same noise, same order of operations.
// Each track and each axis is an independent 2-state filter: [position, velocity].
// tests/js/kalman.test.mjs checks it against the Python smoother on the demo clips.

// obs: Float64Array of length T * N * 2 (metres), NaN where the track is not observed.
// Returns the smoothed position mean and variance, each a Float64Array of length T * N * 2.
export function kalmanSmooth(obs, T, N, fps, accelStd = 3.0, obsStd = 0.3) {
  const dt = 1 / fps;
  const q00 = accelStd ** 2 * dt ** 4 / 4, q01 = accelStd ** 2 * dt ** 3 / 2, q11 = accelStd ** 2 * dt ** 2;
  const r = obsStd ** 2;
  const mean = new Float64Array(T * N * 2), variance = new Float64Array(T * N * 2);
  // filtered and predicted state of one track and axis, for all frames
  const xf0 = new Float64Array(T), xf1 = new Float64Array(T);
  const pf00 = new Float64Array(T), pf01 = new Float64Array(T), pf11 = new Float64Array(T);
  const xp0 = new Float64Array(T), xp1 = new Float64Array(T);
  const pp00 = new Float64Array(T), pp01 = new Float64Array(T), pp11 = new Float64Array(T);

  for (let n = 0; n < N; n++) {
    // first frame where the track is observed (both axes), as in Python: visible.argmax, else 0
    let first = 0;
    for (let k = 0; k < T; k++) {
      const i = (k * N + n) * 2;
      if (!Number.isNaN(obs[i]) && !Number.isNaN(obs[i + 1])) { first = k; break; }
    }
    for (let a = 0; a < 2; a++) {
      const z0 = obs[(first * N + n) * 2 + a];
      let x0 = Number.isNaN(z0) ? 0 : z0, x1 = 0;
      let p00 = 25, p01 = 0, p11 = 25;
      for (let k = 0; k < T; k++) {
        // predict: x = F x, P = F P F' + Q, with F = [[1, dt], [0, 1]]
        const y0 = x0 + dt * x1, y1 = x1;
        const s00 = p00 + 2 * dt * p01 + dt * dt * p11 + q00;
        const s01 = p01 + dt * p11 + q01;
        const s11 = p11 + q11;
        xp0[k] = y0; xp1[k] = y1; pp00[k] = s00; pp01[k] = s01; pp11[k] = s11;
        const i = (k * N + n) * 2;
        // Python: the player must be visible (both axes finite) and this axis finite
        const has = !Number.isNaN(obs[i]) && !Number.isNaN(obs[i + 1]);
        if (has) {
          const s = s00 + r;
          const g0 = s00 / s, g1 = s01 / s;
          const innov = obs[i + a] - y0;
          x0 = y0 + g0 * innov; x1 = y1 + g1 * innov;
          p00 = s00 - g0 * s00; p01 = s01 - g0 * s01; p11 = s11 - g1 * s01;
        } else {
          x0 = y0; x1 = y1; p00 = s00; p01 = s01; p11 = s11;
        }
        xf0[k] = x0; xf1[k] = x1; pf00[k] = p00; pf01[k] = p01; pf11[k] = p11;
      }
      // backward pass: C = Pf[k] F' inv(Pp[k+1])
      let m0 = xf0[T - 1], m1 = xf1[T - 1], v00 = pf00[T - 1], v01 = pf01[T - 1], v11 = pf11[T - 1];
      mean[((T - 1) * N + n) * 2 + a] = m0; variance[((T - 1) * N + n) * 2 + a] = v00;
      for (let k = T - 2; k >= 0; k--) {
        // Pf F' = [[p00 + dt p01, p01], [p01 + dt p11, p11]]
        const b00 = pf00[k] + dt * pf01[k], b01 = pf01[k], b10 = pf01[k] + dt * pf11[k], b11 = pf11[k];
        const a00 = pp00[k + 1], a01 = pp01[k + 1], a11 = pp11[k + 1];
        const det = a00 * a11 - a01 * a01;
        const i00 = a11 / det, i01 = -a01 / det, i11 = a00 / det;
        const c00 = b00 * i00 + b01 * i01, c01 = b00 * i01 + b01 * i11;
        const c10 = b10 * i00 + b11 * i01, c11 = b10 * i01 + b11 * i11;
        const d0 = m0 - xp0[k + 1], d1 = m1 - xp1[k + 1];
        const e00 = v00 - a00, e01 = v01 - a01, e11 = v11 - a11;
        m0 = xf0[k] + c00 * d0 + c01 * d1;
        m1 = xf1[k] + c10 * d0 + c11 * d1;
        // P = Pf + C E C'
        const ce00 = c00 * e00 + c01 * e01, ce01 = c00 * e01 + c01 * e11;
        const ce10 = c10 * e00 + c11 * e01, ce11 = c10 * e01 + c11 * e11;
        v00 = pf00[k] + ce00 * c00 + ce01 * c01;
        v01 = pf01[k] + ce00 * c10 + ce01 * c11;
        v11 = pf11[k] + ce10 * c10 + ce11 * c11;
        mean[(k * N + n) * 2 + a] = m0; variance[(k * N + n) * 2 + a] = v00;
      }
    }
  }
  return { mean, variance };
}
