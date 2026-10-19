// Scripted drone test scenarios for the /sims demo page.
// Everything is hardcoded and deterministic: seeded PRNGs, fixed scripts, no learning or AI.
// The controller mirrors drone/follow.py (KP 90 / KD 60 % stick, stick shaping, 0.25 m/s target that
// waits when the drone lags 0.25 m, heading probe, land after 3 s without a fix). The drone itself is a
// simple point-mass model with stick lag, drag, trim drift and turbulence. Frame: Z up, metres.

export const K = {
  G: 9.81, KP: 90, KD: 60, SPEED: 0.25, LAG_MAX: 0.25, LOST_AFTER: 3.0, FLOOR_CLEARANCE: 0.30,
  MIN_STICK: 12, MAX_STICK: 45, MAX_LIFT: 60, DEAD_ZONE: 0.04,
  PROBE_SETTLE: 1.0, PROBE_PUSH: 0.7, PROBE_STICK: 35, PROBE_COAST: 1.0,
  AMAX_H: 2.6, AMAX_V: 3.0, IGNORE: 10, TAU: 0.12, DRAG: 0.35, R: 0.17, Z0: 0.03,
  DT: 1 / 200, REC: 4, // physics 200 Hz, recorded at 50 Hz
};
const { G, KP, KD, Z0, R } = K;

// --- small math -------------------------------------------------------------
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const norm = (a) => { const l = len(a) || 1; return scl(a, 1 / l); };
const lerp = (a, b, t) => add(a, scl(sub(b, a), t));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const rotZ = (a, ang) => { const c = Math.cos(ang), s = Math.sin(ang); return [a[0] * c - a[1] * s, a[0] * s + a[1] * c, a[2]]; };
const deg = (r) => (r * 180 / Math.PI);
const D2R = Math.PI / 180;

// quaternions [x, y, z, w]
const qmul = (a, b) => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
const qnorm = (q) => { const l = Math.hypot(...q) || 1; return q.map((v) => v / l); };
const qaxis = (ax, ang) => { const s = Math.sin(ang / 2); return [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(ang / 2)]; };
const qfromUnit = (u, v) => {
  const r = dot(u, v) + 1;
  if (r < 1e-6) return [1, 0, 0, 0];
  const c = cross(u, v);
  return qnorm([c[0], c[1], c[2], r]);
};
export const qrot = (q, v) => {
  const u = [q[0], q[1], q[2]], t = scl(cross(u, v), 2);
  return add(add(v, scl(t, q[3])), cross(u, t));
};
const qint = (q, w, dt) => {
  const d = qmul([w[0], w[1], w[2], 0], q);
  return qnorm(q.map((v, i) => v + 0.5 * dt * d[i]));
};
export const nlerp = (a, b, t) => {
  const s = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]) < 0 ? -1 : 1;
  return qnorm(a.map((v, i) => v * (1 - t) + b[i] * s * t));
};

export function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// --- paths -----------------------------------------------------------------
export function polyline(points, step = 0.02) {
  const cum = [0];
  for (let i = 1; i < points.length; i++) cum.push(cum[i - 1] + dist(points[i - 1], points[i]));
  const L = cum[cum.length - 1];
  const raw = (s) => {
    s = clamp(s, 0, L);
    let i = 1;
    while (i < cum.length - 1 && cum[i] < s) i++;
    const seg = cum[i] - cum[i - 1] || 1;
    return lerp(points[i - 1], points[i], (s - cum[i - 1]) / seg);
  };
  const pts = [];
  for (let s = 0; s < L; s += step) pts.push(raw(s));
  pts.push(points[points.length - 1].slice());
  return {
    pts, L,
    at: (s) => raw(s),
    dist(p) {
      let best = Infinity;
      for (let i = 1; i < pts.length; i += 1) {
        const a = pts[i - 1], ab = sub(pts[i], a), t = clamp(dot(sub(p, a), ab) / (dot(ab, ab) || 1), 0, 1);
        best = Math.min(best, dist(p, add(a, scl(ab, t))));
      }
      return best;
    },
  };
}

// A throw's arc (what the beacon flew) and the drone's version of it (floored above the landing spot)
export function throwArc(release, yawDeg, D, H, n = 60) {
  const dir = [-Math.sin(yawDeg * D2R), Math.cos(yawDeg * D2R), 0];
  const beacon = [], path = [];
  for (let i = 0; i <= n; i++) {
    const u = i / n, z = release[2] * (1 - u) + 4 * H * u * (1 - u);
    const p = [release[0] + dir[0] * u * D, release[1] + dir[1] * u * D, Math.max(z, 0.02)];
    beacon.push(p);
    path.push([p[0], p[1], Math.max(z, K.FLOOR_CLEARANCE)]);
  }
  const vz0 = Math.sqrt(2 * G * Math.max(0.05, H + release[2] * 0.25));
  return { beacon, path, dir, flight: clamp(2 * vz0 / G * 1.05, 0.5, 1.4) };
}

// --- simulation state ------------------------------------------------------
export function log(S, msg, lvl = 'info') { S.log.push({ t: S.t, msg, lvl }); }

function shape(v, cap) { return Math.abs(v) < 1e-6 ? 0 : Math.sign(v) * Math.min(cap, K.MIN_STICK + Math.abs(v)); }
// Toy drones ignore the bottom of the stick range: response starts above IGNORE %
const response = (s) => Math.sign(s) * Math.max(0, Math.abs(s) - K.IGNORE);

function crash(S, why, imp, at) {
  if (S.crashed) return;
  const g = S.gauss;
  S.crashed = true; S.armed = false; S.tracking = false; S.queue = []; S.step = null; S.override = null;
  S.impact = imp; S.mode = S.expectCrash ? 'E-STOP · IMPACT' : 'CRASHED';
  S.w = add(S.w, [g() * (2 + imp * 5), g() * (2 + imp * 5), g() * (1 + imp * 3)]);
  log(S, `${why} · ${imp.toFixed(2)} m/s${S.eStop ? '' : ' · MOTORS CUT'}`, S.expectCrash ? 'warn' : 'bad');
  S.sparks.push({ t: S.t, p: at, imp, seed: S.sparks.length + 7 });
  S.doneAt = S.t + 3.5;
}

function makeSim(sc, seed, lite) {
  const rnd = mulberry32(seed);
  const gauss = () => { const u = Math.max(1e-9, rnd()), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const start = sc.start ? [sc.start[0], sc.start[1], Z0] : [0, 0, Z0];
  const trimMag = sc.trim ?? 0.05, trimDir = rnd() * Math.PI * 2;
  return {
    sc, rnd, gauss, lite, t: 0, home: start.slice(),
    p: start.slice(), v: [0, 0, 0], a: [0, 0, 0], est: start.slice(), vest: [0, 0, 0],
    q: qaxis([0, 0, 1], (sc.yaw || 0) * D2R), w: [0, 0, 0], yaw: (sc.yaw || 0) * D2R, yawEst: 0,
    armed: false, crashed: false, onFloor: true, landed: false, mode: 'PREFLIGHT',
    battery: sc.battery ?? 100, drain: sc.drain ?? 0.24,
    fixOK: true, fixAge: 0, goal: null, override: null, stick: { r: 0, p: 0, t: 0 }, motors: [0, 0, 0, 0],
    wind: [0, 0, 0], turb: [0, 0, 0], windFn: null, trim: [Math.cos(trimDir) * trimMag, Math.sin(trimDir) * trimMag, 0],
    motorFail: -1, failT: 0, queue: [], step: null, log: [], frames: [], hist: [],
    doneAt: null, tracking: false, path: null, errSum: 0, errN: 0, errMax: 0, lastErr: 0,
    warn: null, warnOn: false, sparks: [], obstacles: sc.obstacles || [],
    failsafe: false, rth: false, eStop: false, expectCrash: false, impact: 0, flags: {},
  };
}

// --- flight steps (a scenario is a queue of these) ---------------------------
export const step = {
  preflight: (d = 1.6) => ({
    mode: 'PREFLIGHT',
    enter(S) { log(S, 'LINK OK · video 20 fps · variant B'); this.cal = false; },
    run(S) {
      if (!this.cal && S.t - this.t0 > d * 0.5) { this.cal = true; log(S, 'GYRO CALIBRATED · trims loaded'); }
      return S.t - this.t0 >= d;
    },
  }),
  takeoff: (z = 1.0) => ({
    mode: 'TAKEOFF',
    enter(S) { S.armed = true; S.onFloor = false; S.landed = false; this.g = S.p.slice(); S.goal = this.g.slice(); log(S, `ARM · TAKEOFF → ${z.toFixed(2)} m`); },
    run(S, dt) {
      this.g[2] = Math.min(z, this.g[2] + 0.6 * dt);
      S.goal = this.g.slice();
      return this.g[2] >= z && Math.abs(S.p[2] - z) < 0.05 && len(S.v) < 0.12;
    },
  }),
  hold: (d, mode = 'HOLD', track = false) => ({
    mode,
    enter(S) { this.g = (S.goal || S.est).slice(); if (track) { S.tracking = true; S.path = null; log(S, `${mode} · ${d}s at ${this.g.map((v) => v.toFixed(2)).join(', ')}`); } },
    run(S) {
      S.goal = this.g.slice();
      const done = S.t - this.t0 >= d;
      if (done && track) { S.tracking = false; log(S, `HOLD DONE · mean drift ${(S.errSum / Math.max(1, S.errN) * 100).toFixed(1)} cm`, 'ok'); }
      return done;
    },
  }),
  moveTo: (target, speed = 0.4, mode = 'TRANSIT', msg) => ({
    mode,
    enter(S) { this.g = (S.goal || S.est).slice(); if (msg) log(S, msg); },
    run(S, dt) {
      const d = sub(target, this.g), l = len(d), stepL = speed * dt;
      this.g = l <= stepL ? target.slice() : add(this.g, scl(d, stepL / l));
      S.goal = this.g.slice();
      return l <= stepL && dist(S.p, target) < 0.08;
    },
  }),
  follow: (points, { speed = K.SPEED, mode = 'FOLLOW PATH', label = 'path' } = {}) => ({
    mode,
    enter(S) {
      this.path = polyline(points); this.s = 0;
      S.tracking = true; S.path = this.path;
      log(S, `FOLLOW ${label} · ${this.path.L.toFixed(2)} m @ ${speed.toFixed(2)} m/s`);
    },
    run(S, dt) {
      if (S.fixOK && dist(S.est, this.path.at(this.s)) < K.LAG_MAX) this.s = Math.min(this.path.L, this.s + speed * dt);
      S.goal = this.path.at(this.s);
      const done = this.s >= this.path.L && dist(S.p, S.goal) < 0.06;
      if (done) {
        S.tracking = false;
        log(S, `PATH COMPLETE · mean ${(S.errSum / Math.max(1, S.errN) * 100).toFixed(1)} cm · max ${(S.errMax * 100).toFixed(1)} cm`, 'ok');
      }
      return done;
    },
  }),
  // Like follow.py: settle, push forward, coast, then read which way it went
  probe: () => ({
    mode: 'HEADING PROBE',
    enter(S) { this.z = S.goal ? S.goal[2] : S.p[2]; this.ph = 0; log(S, `HEADING PROBE · ${K.PROBE_STICK}% fwd for ${K.PROBE_PUSH}s`); },
    run(S) {
      const el = S.t - this.t0;
      if (el < K.PROBE_SETTLE) { this.from = S.est.slice(); S.goal = [this.from[0], this.from[1], this.z]; return false; }
      if (el < K.PROBE_SETTLE + K.PROBE_PUSH) { S.override = { x: 0, y: K.PROBE_STICK }; S.goal = [S.est[0], S.est[1], this.z]; return false; }
      if (el < K.PROBE_SETTLE + K.PROBE_PUSH + K.PROBE_COAST) { S.override = { x: 0, y: 0 }; S.goal = [S.est[0], S.est[1], this.z]; return false; }
      if (this.ph === 0) {
        this.ph = 1; S.override = null;
        const d = sub(S.est, this.from);
        S.yawEst = Math.atan2(d[1], d[0]) - Math.PI / 2;
        const err = ((deg(S.yawEst - S.yaw) + 540) % 360) - 180;
        log(S, `HEADING = ${deg(S.yawEst).toFixed(1)}° · moved ${(Math.hypot(d[0], d[1]) * 100).toFixed(0)} cm · err ${err.toFixed(1)}°`, 'ok');
        S.goal = [S.est[0], S.est[1], this.z];
      }
      return len(S.v) < 0.06;
    },
  }),
  land: (mode = 'LANDING') => ({
    mode,
    enter(S) { this.g = (S.goal || S.est).slice(); log(S, `${mode}`); },
    run(S, dt) {
      if (S.fixOK) { this.g[2] = Math.max(Z0 - 0.15, this.g[2] - 0.35 * dt); S.goal = this.g.slice(); S.override = null; }
      else S.override = { z: clamp(70 * (-0.3 - S.v[2]), -40, 40) }; // drone's own land command, no fix needed
      if (S.p[2] <= Z0 + 0.01) {
        S.armed = false; S.onFloor = true; S.landed = true; S.p[2] = Z0; S.v = [0, 0, 0]; S.a = [0, 0, 0];
        S.mode = 'LANDED'; log(S, 'TOUCHDOWN · DISARM', 'ok');
        return true;
      }
      return false;
    },
  }),
  complete: () => ({
    mode: 'COMPLETE',
    enter(S) { if (!S.armed) S.mode = 'LANDED · COMPLETE'; },
    run(S) { S.doneAt ??= S.t + 1.5; return false; },
  }),
};

// --- physics ---------------------------------------------------------------
function physics(S, dt) {
  const g = S.gauss;
  for (let k = 0; k < 3; k++) S.turb[k] += (g() * 2.2 - S.turb[k]) * Math.min(1, dt * 1.2);
  const gust = S.windFn ? S.windFn(S.t) : [0, 0, 0];
  S.wind = add([S.turb[0] * 0.05, S.turb[1] * 0.05, S.turb[2] * 0.02], gust);

  // Camera + LiDAR fix: ~80 ms late, ~1 cm noise
  S.hist.push(S.p.slice()); if (S.hist.length > 16) S.hist.shift();
  if (S.fixOK) {
    const d = S.hist[0];
    S.est = [d[0] + g() * 0.008, d[1] + g() * 0.008, d[2] + g() * 0.006];
    S.vest = lerp(S.vest, [S.v[0] + g() * 0.04, S.v[1] + g() * 0.04, S.v[2] + g() * 0.03], 0.3);
    S.fixAge = 0;
  } else S.fixAge += dt;

  if (S.armed && !S.fixOK && S.fixAge >= K.LOST_AFTER && !S.failsafe && !S.sc.noFailsafe) {
    S.failsafe = true; S.tracking = false;
    log(S, `NO FIX ${K.LOST_AFTER.toFixed(1)} s → FAILSAFE LAND`, 'warn');
    S.queue = [step.land('FAILSAFE LAND'), step.complete()]; S.step = null;
  }

  if (S.onFloor && !S.armed && !S.crashed) { S.motors = [0, 0, 0, 0]; S.stick = { r: 0, p: 0, t: 0 }; return; }

  if (S.armed && !S.crashed) {
    let u = [0, 0, 0];
    if (S.fixOK && S.goal) {
      const e = sub(S.goal, S.est);
      const push = rotZ([KP * e[0] - KD * S.vest[0], KP * e[1] - KD * S.vest[1], 0], -S.yawEst);
      const flat = Math.hypot(e[0], e[1]) > K.DEAD_ZONE;
      u = [flat ? shape(push[0], K.MAX_STICK) : 0, flat ? shape(push[1], K.MAX_STICK) : 0,
        Math.abs(e[2]) > K.DEAD_ZONE ? shape(KP * e[2] - KD * S.vest[2], K.MAX_LIFT) : 0];
    }
    const o = S.override;
    if (o) { if (o.x != null) { u[0] = o.x; u[1] = o.y; } if (o.z != null) u[2] = o.z; }
    S.stick = { r: u[0], p: u[1], t: u[2] };

    let cmd = rotZ([response(u[0]) * K.AMAX_H / 100, response(u[1]) * K.AMAX_H / 100, response(u[2]) * K.AMAX_V / 100], S.yaw);
    let arm = 0;
    if (S.motorFail >= 0) {
      const el = S.t - S.failT;
      S.yaw += 10 * dt * Math.min(1, el * 1.5);
      arm = S.yaw + S.motorFail * Math.PI / 2 + Math.PI / 4;
      cmd = [cmd[0] * 0.3 + 2.2 * Math.cos(arm), cmd[1] * 0.3 + 2.2 * Math.sin(arm), cmd[2] * 0.3 - 4.8];
    }
    S.a = lerp(S.a, cmd, Math.min(1, dt / K.TAU));
    const acc = [S.a[0] + S.wind[0] + S.trim[0] - K.DRAG * S.v[0], S.a[1] + S.wind[1] + S.trim[1] - K.DRAG * S.v[1], S.a[2] + S.wind[2] - K.DRAG * S.v[2]];
    S.v = add(S.v, scl(acc, dt));
    S.p = add(S.p, scl(S.v, dt));

    const thrust = norm([S.a[0], S.a[1], G + S.a[2]]);
    S.q = qmul(qfromUnit([0, 0, 1], thrust), qaxis([0, 0, 1], S.yaw));
    if (S.motorFail >= 0) S.q = qmul(qaxis([Math.cos(arm + Math.PI / 2), Math.sin(arm + Math.PI / 2), 0], -0.45), S.q);

    // Motor mix: FL, FR, RR, RL
    const base = clamp(0.52 + u[2] / 100 * 0.4, 0, 1), r = u[0] / 100 * 0.35, p = u[1] / 100 * 0.35;
    S.motors = [base + r - p, base - r - p, base - r + p, base + r + p].map((m, i) => (i === S.motorFail ? 0 : clamp(m, 0, 1)));
    if (S.motorFail >= 0) S.motors = S.motors.map((m, i) => (i === S.motorFail ? 0 : 1));

    S.battery = Math.max(0, S.battery - S.drain * dt * (0.85 + Math.hypot(u[0], u[1], u[2]) / 150));
    if (S.battery < 20 && !S.flags.lowWarn) { S.flags.lowWarn = true; log(S, `LOW BATTERY ${S.battery.toFixed(0)}%`, 'warn'); }
    if (S.battery < 15 && !S.rth && !S.sc.noRTH) {
      S.rth = true; S.tracking = false;
      log(S, 'BATTERY 15% → ABORT · RETURN HOME + LAND', 'warn');
      S.queue = [step.moveTo([S.home[0], S.home[1], Math.max(0.6, S.p[2])], 0.45, 'LOW BATT · RTH'), step.land('LOW BATT · LANDING'), step.complete()];
      S.step = null;
    }

    if (S.p[2] < Z0) {
      if (S.v[2] < -1.0) { S.p[2] = Z0; crash(S, 'HARD IMPACT · FLOOR', -S.v[2], [S.p[0], S.p[1], 0]); }
      else { S.p[2] = Z0; S.v[2] = Math.max(0, S.v[2]); S.a[2] = Math.max(0, S.a[2]); }
    }
  } else {
    // Unpowered: falls and tumbles, bounces on the floor, comes to rest
    S.motors = [0, 0, 0, 0]; S.stick = { r: 0, p: 0, t: 0 };
    S.v[2] -= G * dt;
    S.v = scl(S.v, 1 - 0.25 * dt);
    S.p = add(S.p, scl(S.v, dt));
    S.q = qint(S.q, S.w, dt);
    if (S.p[2] < Z0) {
      S.p[2] = Z0;
      if (S.v[2] < -0.4) {
        const imp = -S.v[2];
        if (!S.crashed) crash(S, S.eStop ? 'IMPACT AFTER E-STOP' : 'IMPACT · FLOOR', imp, [S.p[0], S.p[1], 0]);
        S.v = [S.v[0] * 0.55, S.v[1] * 0.55, imp * 0.28];
        S.w = add(scl(S.w, 0.5), [g() * imp * 2, g() * imp * 2, g() * imp]);
      } else {
        S.v[2] = 0;
        const f = Math.exp(-5 * dt);
        S.v[0] *= f; S.v[1] *= f;
        S.w = scl(S.w, Math.exp(-7 * dt));
        const up = qrot(S.q, [0, 0, 1]);
        const target = qmul(qfromUnit(up, up[2] >= 0 ? [0, 0, 1] : [0, 0, -1]), S.q);
        S.q = nlerp(S.q, target, 1 - Math.exp(-10 * dt));
      }
    }
  }

  // Obstacles: push out, bounce, crash if it hit with any real speed
  for (const ob of S.obstacles) {
    const c = [clamp(S.p[0], ob.min[0], ob.max[0]), clamp(S.p[1], ob.min[1], ob.max[1]), clamp(S.p[2], ob.min[2], ob.max[2])];
    const d = sub(S.p, c), l = len(d);
    if (l < R) {
      const n = l > 1e-6 ? scl(d, 1 / l) : [0, 0, 1];
      S.p = add(c, scl(n, R));
      const vn = dot(S.v, n);
      if (vn < 0) {
        S.v = sub(S.v, scl(n, 1.45 * vn));
        if (!S.crashed && -vn > 0.12) crash(S, `COLLISION · ${ob.label}`, -vn, c);
      }
    }
  }

  // Obstacle warning (warn only, like drone/obstacles.py): close, or reached in under 2 s
  S.warn = null;
  if (S.armed) {
    for (const ob of S.obstacles) {
      const c = [clamp(S.p[0], ob.min[0], ob.max[0]), clamp(S.p[1], ob.min[1], ob.max[1]), clamp(S.p[2], ob.min[2], ob.max[2])];
      const l = dist(S.p, c), d = Math.max(0, l - R), closing = l > 1e-6 ? dot(S.v, scl(sub(c, S.p), 1 / l)) : 0;
      const ttc = closing > 0.05 ? d / closing : Infinity;
      if (d < 0.25 || ttc < 2) { if (!S.warn || d < S.warn.d) S.warn = { label: ob.label, d, ttc }; }
    }
  }
  if (S.warn && !S.warnOn) log(S, `OBSTACLE AHEAD · ${S.warn.label} ${S.warn.d.toFixed(2)} m · TTC ${isFinite(S.warn.ttc) ? S.warn.ttc.toFixed(1) + ' s' : '--'} (warning only)`, 'warn');
  S.warnOn = !!S.warn;
}

function record(S) {
  let err = null;
  if (S.tracking) {
    err = S.path ? S.path.dist(S.p) : dist(S.p, S.goal || S.p);
    S.errSum += err; S.errN++; S.errMax = Math.max(S.errMax, err); S.lastErr = err;
  }
  if (S.lite) { S.frames.push([S.p[0], S.p[1], S.p[2], S.crashed ? 1 : 0]); return; }
  const g = S.gauss;
  S.frames.push({
    t: S.t, p: S.p.slice(), q: S.q.slice(), v: S.v.slice(), a: S.a.slice(),
    goal: S.goal && S.armed ? S.goal.slice() : null, est: S.est.slice(), fixOK: S.fixOK, fixAge: S.fixAge,
    mode: S.mode, stick: { ...S.stick }, motors: S.motors.slice(), bat: S.battery,
    err, errMean: S.errN ? S.errSum / S.errN : null, errMax: S.errN ? S.errMax : null,
    warn: S.warn ? { ...S.warn } : null, armed: S.armed, crashed: S.crashed,
    wind: Math.hypot(S.wind[0], S.wind[1]), gust: S.windFn ? Math.hypot(...S.windFn(S.t)) : 0,
    rssi: -46 - 4 * Math.hypot(S.p[0], S.p[1]) + 2 * Math.sin(S.t * 0.7) + g(),
    yaw: S.yaw, yawEst: S.yawEst, motorFail: S.motorFail, eStop: S.eStop,
  });
}

function verdict(S) {
  const th = S.sc.thresh || { mean: 0.06, max: 0.25 };
  const mean = S.errN ? S.errSum / S.errN : null, max = S.errN ? S.errMax : null;
  if (S.crashed) return S.expectCrash ? { txt: 'PASS · E-STOP', cls: 'warn', ok: true } : { txt: 'FAIL · CRASH', cls: 'bad', ok: false };
  if (!S.landed) return { txt: 'INCOMPLETE', cls: 'bad', ok: false };
  if (S.failsafe) return { txt: 'PASS · FAILSAFE LAND', cls: 'warn', ok: true };
  if (S.rth) return { txt: 'PASS · AUTO RTH', cls: 'warn', ok: true };
  if (mean != null && (mean > th.mean || max > th.max)) return { txt: 'FAIL · OFF PATH', cls: 'bad', ok: false };
  return { txt: 'PASS', cls: 'ok', ok: true };
}

export function run(sc, { seed = sc.seed ?? 1, lite = false, params } = {}) {
  const S = makeSim(params ? { ...sc, ...params } : sc, seed, lite);
  const plan = (params?.build || sc.build)(S);
  S.queue = plan.queue;
  const events = (plan.events || []).map((e) => ({ ...e, fired: false }));
  const max = sc.max ?? 60;
  let n = 0;
  while (S.t <= max) {
    for (const e of events) {
      if (e.fired) continue;
      if ((e.t != null && S.t >= e.t) || (e.when && e.when(S))) { e.fired = true; e.fn(S); }
    }
    if (!S.step && S.queue.length && !S.crashed) {
      S.step = S.queue.shift(); S.step.t0 = S.t; S.override = null;
      S.mode = S.step.mode; S.step.enter?.(S);
    }
    if (S.step && !S.crashed && S.step.run(S, K.DT)) S.step = null;
    physics(S, K.DT);
    if (n % K.REC === 0) record(S);
    n++; S.t += K.DT;
    if (S.doneAt != null && S.t > S.doneAt) break;
  }
  record(S);
  S.duration = S.t;
  S.verdict = verdict(S);
  S.errMeanFinal = S.errN ? S.errSum / S.errN : null;
  S.errMaxFinal = S.errN ? S.errMax : null;
  S.hist = null;
  return S;
}

// --- scenarios -------------------------------------------------------------
const box = (min, max, label, kind = 'obstacle') => ({ min, max, label, kind });
const chairAt = (x, y, facingDeg = 0) => {
  // Seat block + backrest on the far side (axis-aligned; facing 0 = back toward +Y)
  const back = facingDeg === 0 ? [y + 0.17, y + 0.22] : [y - 0.22, y - 0.17];
  return [box([x - 0.22, y - 0.22, 0], [x + 0.22, y + 0.22, 0.46], 'CHAIR', 'chair'),
    box([x - 0.22, back[0], 0.46], [x + 0.22, back[1], 0.92], 'CHAIR', 'chair')];
};
const roomWalls = (h = 2.3, H = 2.2) => [
  box([-h - 0.1, -h - 0.1, 0], [h + 0.1, -h, H], 'WALL S', 'wall'), box([-h - 0.1, h, 0], [h + 0.1, h + 0.1, H], 'WALL N', 'wall'),
  box([-h - 0.1, -h, 0], [-h, h, H], 'WALL W', 'wall'), box([h, -h, 0], [h + 0.1, h, H], 'WALL E', 'wall')];

function throwScenario(release, yaw, D, H, extra = {}) {
  const arc = throwArc(release, yaw, D, H);
  const pad = [release[0] - arc.dir[0] * 0.5, release[1] - arc.dir[1] * 0.5];
  return {
    start: pad, arc, ref: [arc.path],
    beacon: { pts: arc.beacon, t0: 0.4, dur: arc.flight },
    build: (S) => ({
      queue: [step.preflight(2.4), step.takeoff(arc.path[0][2]), step.moveTo(arc.path[0], 0.3, 'TO PATH START'),
        step.follow(arc.path, { label: `throw #${extra.pathId ?? 12}` }), step.hold(1.0), step.land(), step.complete()],
      events: [
        { t: 0.4, fn: (S) => log(S, 'VAE1 launch · beacon airborne') },
        { t: 0.4 + arc.flight, fn: (S) => log(S, `VAE1 land · path #${extra.pathId ?? 12} captured (${D.toFixed(2)} m, apex ${(H + release[2] * 0.5).toFixed(2)} m)`, 'ok') },
        ...(extra.events || []),
      ],
    }),
  };
}

const figure8 = (a = 1.0, b = 0.55, z = 1.0, dz = 0.2, cx = 0, cy = 0, loops = 1) => {
  const pts = [];
  for (let i = 0; i <= 120 * loops; i++) {
    const th = i / 120 * Math.PI * 2;
    pts.push([cx + a * Math.sin(th), cy + b * Math.sin(2 * th), z + dz * Math.sin(th)]);
  }
  return pts;
};
const gustFn = (t0, dur, amp, dirDeg) => (t) => {
  if (t < t0 || t > t0 + dur) return [0, 0, 0];
  const s = Math.sin(Math.PI * (t - t0) / dur) ** 2 * amp;
  return [-Math.sin(dirDeg * D2R) * s, Math.cos(dirDeg * D2R) * s, 0];
};
const inFollowFor = (sec) => (S) => S.step && S.step.mode === 'FOLLOW PATH' && S.t - S.step.t0 > sec;

const BOX_PATH = [[-0.6, -0.6, 1.0], [0.6, -0.6, 1.0], [0.6, 0.6, 1.3], [-0.6, 0.6, 1.3], [-0.6, -0.6, 1.0]];
const LINE_X = [[-1.2, 0, 1.0], [1.2, 0, 1.0]];
const LINE_DIAG = [[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0]];

export const SCENARIOS = [
  {
    id: 'T01', name: 'Takeoff · Hover hold · Land', cat: 'NOMINAL', seed: 11,
    desc: 'Baseline: lift to 1.0 m and hold position for 8 s on the camera + LiDAR fix, then land on the pad.',
    start: [0, 0], thresh: { mean: 0.05, max: 0.12 },
    build: () => ({ queue: [step.preflight(), step.takeoff(1.0), step.hold(8, 'POSITION HOLD', true), step.land(), step.complete()] }),
  },
  {
    id: 'T02', name: 'Throw follow · short toss', cat: 'NOMINAL', seed: 21,
    desc: 'Beacon is tossed 1.6 m. The drone flies the same arc at 0.25 m/s, floored 30 cm above the landing spot.',
    ...throwScenario([0, -0.8, 1.0], 0, 1.6, 0.35, { pathId: 12 }),
  },
  {
    id: 'T03', name: 'Throw follow · long lob', cat: 'NOMINAL', seed: 31,
    desc: 'High 2.6 m lob thrown off-axis (-25°). Tests the climb to the apex and the descent toward the floor clamp.',
    ...throwScenario([-0.6, -1.2, 1.1], -25, 2.6, 0.8, { pathId: 17 }),
  },
  {
    id: 'T04', name: '3D box pattern', cat: 'MANEUVER', seed: 41,
    desc: '1.2 m square with a 30 cm climb on the far side. Sharp corners show how much the PD loop cuts them.',
    start: [-0.6, -0.6], ref: [BOX_PATH], thresh: { mean: 0.06, max: 0.2 },
    build: () => ({ queue: [step.preflight(), step.takeoff(1.0), step.follow(BOX_PATH, { speed: 0.3, label: 'box' }), step.hold(1), step.land(), step.complete()] }),
  },
  {
    id: 'T05', name: 'Figure-8 with altitude wave', cat: 'MANEUVER', seed: 51,
    desc: 'Continuous curvature in X, Y and Z at 0.35 m/s. The hardest smooth path in the set.',
    start: [0, 0], ref: [figure8()], thresh: { mean: 0.06, max: 0.2 },
    build: () => ({ queue: [step.preflight(), step.takeoff(1.0), step.follow(figure8(), { speed: 0.35, label: 'figure-8' }), step.hold(1), step.land(), step.complete()] }),
  },
  {
    id: 'T06', name: 'Heading probe · unknown yaw 37°', cat: 'MANEUVER', seed: 61, yaw: 37, trim: 0.02,
    desc: 'The drone has no compass and never yaws. It pushes forward once, measures which way it moved, then flies a triangle.',
    start: [0, -0.9],
    ref: [[[0, -0.3, 1.0], [0.9, 0.7, 1.2], [-0.9, 0.7, 1.2], [0, -0.3, 1.0]]],
    build: (S) => {
      const tri = S.sc.ref[0];
      return { queue: [step.preflight(), step.takeoff(1.0), step.probe(), step.moveTo(tri[0], 0.3, 'TO PATH START'), step.follow(tri, { label: 'triangle' }), step.hold(1), step.land(), step.complete()] };
    },
  },
  {
    id: 'T07', name: 'Gust rejection · fan test', cat: 'ROBUSTNESS', seed: 71,
    desc: 'A 0.7 m/s² side gust (desk fan) hits mid-path. Watch the error spike and the controller pull it back.',
    start: [-1.2, -0.4], ref: [LINE_X], thresh: { mean: 0.08, max: 0.35 },
    build: () => ({
      queue: [step.preflight(), step.takeoff(1.0), step.moveTo(LINE_X[0], 0.3, 'TO PATH START'), step.follow(LINE_X, { label: 'line' }), step.hold(1), step.land(), step.complete()],
      events: [
        { when: inFollowFor(3), fn: (S) => { S.windFn = gustFn(S.t, 1.8, 0.7, 0); S.flags.gustEnd = S.t + 1.8; log(S, 'GUST 0.7 m/s² from -Y (fan on)', 'warn'); } },
        { when: (S) => S.flags.gustEnd && S.t > S.flags.gustEnd && S.lastErr < 0.05, fn: (S) => log(S, `RECOVERED · back under 5 cm, peak ${(S.errMax * 100).toFixed(0)} cm`, 'ok') },
      ],
    }),
  },
  {
    id: 'T08', name: 'Brief occlusion · track reacquired', cat: 'ROBUSTNESS', seed: 81, trim: 0.08,
    desc: 'Someone walks between the phone and the drone for 1.4 s. Sticks go neutral, then the fix returns before the 3 s limit.',
    start: [-1.0, -1.3], ref: [LINE_DIAG], thresh: { mean: 0.08, max: 0.3 },
    build: () => ({
      queue: [step.preflight(), step.takeoff(1.0), step.moveTo(LINE_DIAG[0], 0.3, 'TO PATH START'), step.follow(LINE_DIAG, { label: 'diagonal' }), step.hold(1), step.land(), step.complete()],
      events: [
        { when: inFollowFor(3.5), fn: (S) => { S.fixOK = false; S.flags.lostT = S.t; log(S, 'VISION TRACK LOST · drone occluded · sticks neutral', 'warn'); } },
        { when: (S) => S.flags.lostT && S.t > S.flags.lostT + 1.4, fn: (S) => { S.fixOK = true; log(S, 'TRACK REACQUIRED after 1.4 s · resuming', 'ok'); } },
      ],
    }),
  },
  {
    id: 'T09', name: 'Tracking lost · failsafe landing', cat: 'FAILSAFE', seed: 91, trim: 0.09,
    desc: 'The phone loses the drone for good. After 3.0 s with no fix the follower gives up and the drone lands itself.',
    start: [-1.0, -1.3], ref: [LINE_DIAG],
    build: () => ({
      queue: [step.preflight(), step.takeoff(1.0), step.moveTo(LINE_DIAG[0], 0.3, 'TO PATH START'), step.follow(LINE_DIAG, { label: 'diagonal' }), step.hold(1), step.land(), step.complete()],
      events: [{ when: inFollowFor(3.5), fn: (S) => { S.fixOK = false; log(S, 'VISION TRACK LOST · drone left the camera view', 'warn'); } }],
    }),
  },
  {
    id: 'T10', name: 'Low battery · auto return + land', cat: 'FAILSAFE', seed: 101, battery: 24, drain: 0.55,
    desc: 'Flying a double figure-8 on a weak pack (drain sped up for the demo). At 15 % it aborts, flies home and lands.',
    start: [0, -1.0], ref: [figure8(1.0, 0.55, 1.0, 0.2, 0, 0.2, 2)],
    build: (S) => ({ queue: [step.preflight(), step.takeoff(1.0), step.moveTo(S.sc.ref[0][0], 0.3, 'TO PATH START'), step.follow(S.sc.ref[0], { speed: 0.35, label: 'figure-8 x2' }), step.hold(1), step.land(), step.complete()] }),
  },
  {
    id: 'T11', name: 'CRASH · flown into a wall', cat: 'CRASH', seed: 111,
    desc: 'Manual keyboard flight, W held toward a wall. The obstacle warning fires (warning only, it never steers) and the operator does not brake.',
    lesson: 'Obstacle warning is advisory by design. Impact speed ~0.8 m/s. The operator has to react inside the 2 s TTC window.',
    start: [0, -0.6], obstacles: [box([-2.4, 2.0, 0], [2.4, 2.12, 2.2], 'WALL', 'wall')],
    build: () => ({ queue: [step.preflight(), step.takeoff(1.0), step.moveTo([0, 3.2, 1.0], 0.8, 'MANUAL · W HELD', 'OPERATOR: holding W (forward)'), step.complete()] }),
  },
  {
    id: 'T12', name: 'CRASH · motor 3 failure', cat: 'CRASH', seed: 121,
    desc: 'Rear-right motor stops mid-box (simulated loose connector). Lift drops, the frame spins, and the drone falls.',
    lesson: 'A quad cannot fly on 3 motors. The other three go to full power but it still falls in under a second, so the only mitigation is pre-flight checks and prop guards.',
    start: [-0.6, -0.6], ref: [BOX_PATH],
    build: () => ({
      queue: [step.preflight(), step.takeoff(1.0), step.follow(BOX_PATH, { speed: 0.3, label: 'box' }), step.hold(1), step.land(), step.complete()],
      events: [{ when: inFollowFor(4.5), fn: (S) => { S.motorFail = 2; S.failT = S.t; S.mode = 'MOTOR 3 FAILURE'; log(S, 'MOTOR 3 (RR) · NO RPM · thrust lost', 'bad'); } }],
    }),
  },
  {
    id: 'T13', name: 'CRASH · wrong heading, probe disabled', cat: 'CRASH', seed: 131, yaw: 150,
    desc: 'Same follow, but the heading probe is switched off and the drone actually faces 150° away. Every correction pushes it the wrong way.',
    lesson: 'With the heading wrong by more than 90°, the PD loop pushes away from the path and diverges within seconds. This is why the heading probe runs before every follow (see T06).',
    start: [0, -0.4], obstacles: roomWalls(), ref: [LINE_X],
    build: () => ({
      queue: [step.preflight(), step.takeoff(1.0), step.moveTo(LINE_X[0], 0.3, 'TO PATH START', 'heading probe DISABLED · assuming 0° (actual 150°)'), step.follow(LINE_X, { label: 'line' }), step.hold(1), step.land(), step.complete()],
    }),
  },
  {
    id: 'T14', name: 'CRASH · chair on the landing path', cat: 'CRASH', seed: 141,
    desc: 'The beacon lands just past a chair. The drone path is only floored 30 cm above the landing spot, so it flies into the seat.',
    lesson: 'Path following does not avoid obstacles. The warning fired ~1.6 s before contact. Paths must be clear or the floor clearance raised.',
    ...throwScenario([0, -1.2, 1.0], 0, 2.2, 0.45, { pathId: 23 }),
    obstacles: chairAt(0, 0.65),
  },
  {
    id: 'T15', name: 'Emergency stop mid-air', cat: 'FAILSAFE', seed: 151,
    desc: 'Operator hits SPACE during a figure-8 at 1.2 m. Motors cut instantly; the drone falls. Expected crash, test is that the cut works.',
    start: [0, 0], ref: [figure8(1.0, 0.55, 1.2, 0.15)],
    build: (S) => ({
      queue: [step.preflight(), step.takeoff(1.2), step.follow(S.sc.ref[0], { speed: 0.35, label: 'figure-8' }), step.hold(1), step.land(), step.complete()],
      events: [{
        when: inFollowFor(5), fn: (S) => {
          S.armed = false; S.eStop = true; S.expectCrash = true; S.tracking = false; S.queue = []; S.step = null;
          S.mode = 'E-STOP · MOTORS OFF'; S.w = [0.8, -0.5, 1.5];
          log(S, 'OPERATOR: SPACE → EMERGENCY STOP · motors off', 'warn');
        },
      }],
    }),
  },
];

// Batch: many randomized throws through the same pipeline
export const BATCH = { id: 'T16', name: 'Batch · 60 randomized throws', cat: 'BATCH', count: 60, seed: 1600,
  desc: '60 throws with random distance (1.2-2.8 m), height, direction, drone trim and a random side gust. Same controller, same thresholds.' };

export function runBatch(count = BATCH.count) {
  const out = [];
  for (let i = 0; i < count; i++) {
    const r = mulberry32(BATCH.seed + i * 977);
    const D = 1.2 + r() * 1.6, H = 0.2 + r() * 0.7, z0 = 0.9 + r() * 0.4, yaw = i * (360 / count) + (r() - 0.5) * 4;
    const gustAt = 2 + r() * 4, gustAmp = r() * r() * 2.4, gustDir = r() * 360, trim = r() * 0.07;
    const release = [-Math.sin(yaw * D2R) * 0.35, Math.cos(yaw * D2R) * 0.35, z0];
    const sc = { id: `B${String(i + 1).padStart(2, '0')}`, seed: BATCH.seed + i, trim, max: 60, ...throwScenario(release, yaw, D, H, {
      pathId: 100 + i,
      events: [{ when: inFollowFor(gustAt), fn: (S) => { S.windFn = gustFn(S.t, 1.5, gustAmp, gustDir); } }],
    }) };
    const S = run(sc, { lite: true });
    out.push({ sc, S, D, H, gustAmp, mean: S.errMeanFinal, max: S.errMaxFinal, verdict: S.verdict });
  }
  return out;
}
