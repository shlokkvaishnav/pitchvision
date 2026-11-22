// Pitch control (Spearman 2018) as in Laurie Shaw's LaurieOnTracking (MIT), both teams alike.
// Same algorithm as src/pitchvision/pitch_control.py; tests/js/pitchcontrol.test.mjs checks they agree.
export const PARAMS = { reactionTime: 0.7, vmax: 5, ttiSigma: 0.45, lam: 4.3, ballSpeed: 15, dt: 0.04, maxTime: 10, tol: 0.01, veto: 3 };

export function grid(nx = 32, length = 105, width = 68) {
  const ny = Math.round((nx * width) / length), cells = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++)
    cells.push([((i + 0.5) * length) / nx - length / 2, ((j + 0.5) * width) / ny - width / 2]);
  return { nx, ny, cells };
}

// pos, vel: arrays of [x, y]; team: 0 home, 1 away; ball: [x, y] or null. Returns the home share of control.
export function homeControl(cell, pos, vel, team, ball, p = PARAMS) {
  const ttc = p.veto * Math.log(10) * ((Math.sqrt(3) * p.ttiSigma) / Math.PI + 1 / p.lam);
  const ballT = ball && Number.isFinite(ball[0]) ? Math.hypot(cell[0] - ball[0], cell[1] - ball[1]) / p.ballSpeed : 0;
  const n = pos.length, tau = new Float64Array(n);
  let th = Infinity, ta = Infinity;
  for (let i = 0; i < n; i++) {
    const rx = pos[i][0] + vel[i][0] * p.reactionTime, ry = pos[i][1] + vel[i][1] * p.reactionTime;
    tau[i] = p.reactionTime + Math.hypot(cell[0] - rx, cell[1] - ry) / p.vmax;
    if (team[i] === 0) th = Math.min(th, tau[i]); else ta = Math.min(ta, tau[i]);
  }
  if (!Number.isFinite(th) || !Number.isFinite(ta)) return Number.isFinite(th) ? 1 : 0; // a team with no players controls nothing
  if (th - Math.max(ballT, ta) >= ttc) return 0;
  if (ta - Math.max(ballT, th) >= ttc) return 1;
  const keepTau = [], keepSide = [];
  for (let i = 0; i < n; i++) {
    const best = team[i] === 0 ? th : ta;
    if (tau[i] - best < ttc) { keepTau.push(tau[i]); keepSide.push(team[i]); }
  }
  const m = keepTau.length, ppcf = new Float64Array(m), k = Math.PI / Math.sqrt(3) / p.ttiSigma;
  const steps = Math.ceil((p.maxTime + p.dt) / p.dt);
  for (let s = 0; s < steps; s++) {
    const t = ballT - p.dt + s * p.dt;
    if (t >= ballT + p.maxTime - 1e-12) break;
    let total = 0;
    for (let i = 0; i < m; i++) total += ppcf[i];
    if (total >= 1 - p.tol) break;
    for (let i = 0; i < m; i++) ppcf[i] += (1 - total) * (1 / (1 + Math.exp(-k * (t - keepTau[i])))) * p.lam * p.dt;
  }
  let home = 0, all = 0;
  for (let i = 0; i < m; i++) { all += ppcf[i]; if (keepSide[i] === 0) home += ppcf[i]; }
  return home / Math.max(all, 1e-12);
}

export function controlMap(pos, vel, team, ball, nx = 32, p = PARAMS) {
  const g = grid(nx), out = new Float64Array(g.cells.length);
  g.cells.forEach((c, i) => { out[i] = homeControl(c, pos, vel, team, ball, p); });
  return { ...g, home: out };
}

// Velocity of each track at frame f from positions (frames x n x [x, y]), central difference over +-2 frames.
export function velocities(series, f, fps) {
  const a = Math.max(0, f - 2), b = Math.min(series.length - 1, f + 2), dt = (b - a) / fps;
  return series[f].map((_, i) => (dt > 0 ? [(series[b][i][0] - series[a][i][0]) / dt, (series[b][i][1] - series[a][i][1]) / dt] : [0, 0]));
}
