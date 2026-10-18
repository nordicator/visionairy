// VisionAiry headset: camera passthrough + F-35 style HUD + the ESP32 beacon's tracked position in AR.
//
// No vision tracking. The beacon streams its own dead-reckoned position (see esp32/beacon), and this phone
// dead-reckons its own position the same way, from its motion sensors. Both start at the same point:
// calibration happens with the phone lying on the beacon, so that spot is the origin for both.
//
// Frames:
//   graph frame - the shared one. Origin = calibration spot, Z up, Y = the beacon's Y arrow at calibration
//                 (= the phone's top edge), X = right. Beacon positions arrive in this frame.
//   headset world - the phone's orientation sensor frame (X east-ish, Y north-ish, Z up). The camera view
//                 is computed here. cal.yaw (heading of the graph's +Y in this frame) converts between them.

const HUD = '#39ff6a';
const HUD_DIM = 'rgba(57,255,106,0.45)';
const PATH = '#ffd23f';
const DEG = Math.PI / 180;

const $ = (id) => document.getElementById(id);
const screenCanvas = $('screen');
const screenCtx = screenCanvas.getContext('2d');

// One eye is rendered once, then copied into both halves of the screen.
const eye = document.createElement('canvas');
const eyeCtx = eye.getContext('2d');

const video = document.createElement('video');
video.playsInline = true;
video.muted = true;

const TRAIL_SECONDS = 6;

const settings = { stereo: true, ipd: 0, passFov: 65 };
const state = {
  yaw: 0, pitch: 0, roll: 0,
  gotSensor: false,
  toast: '', toastUntil: 0,
  hintUntil: 0,
  fps: 0,
  beacon: null, // latest heartbeat relayed by the server
};
const cal = { yaw: null, pendingYaw: null }; // heading of the graph frame's +Y in the headset world
const bcn = { p: null, q: null, still: false, air: false, at: 0, trail: [] }; // beacon, graph frame
const phone = { p: [0, 0, 0], v: [0, 0, 0], quietFor: 0, trail: [] }; // this phone's camera, graph frame
let lastThrow = null; // { v, p0, peak, airtime, impactG, spin, p1 }
window.visionairy = { state, settings, cal, bcn, phone }; // poke at it from the dev console

// ---------------------------------------------------------------------------
// Setup screen

const NUMERIC = ['ipd', 'passFov'];
for (const id of NUMERIC) {
  const saved = localStorage.getItem('hs.' + id);
  if (saved !== null) $(id).value = saved;
}
{
  const saved = localStorage.getItem('hs.stereo');
  if (saved !== null) $('stereo').checked = saved === '1';
}

function readSettings() {
  settings.stereo = $('stereo').checked;
  settings.ipd = +$('ipd').value;
  settings.passFov = +$('passFov').value;
  $('ipdVal').textContent = settings.ipd;
  $('passFovVal').textContent = settings.passFov + '°';
  for (const id of NUMERIC) localStorage.setItem('hs.' + id, $(id).value);
  localStorage.setItem('hs.stereo', settings.stereo ? '1' : '0');
}
document.querySelectorAll('#setup input, #setup select').forEach((el) => el.addEventListener('input', readSettings));
readSettings();

$('go').onclick = async () => {
  $('err').textContent = '';
  try {
    // iOS gates motion sensors behind an explicit permission prompt from a user gesture
    if (typeof DeviceOrientationEvent !== 'undefined' && DeviceOrientationEvent.requestPermission) {
      const res = await DeviceOrientationEvent.requestPermission();
      if (res !== 'granted') throw new Error('motion sensor permission denied');
    }
    if (typeof DeviceMotionEvent !== 'undefined' && DeviceMotionEvent.requestPermission) {
      await DeviceMotionEvent.requestPermission().catch(() => {});
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    $('err').textContent = 'Startup failed: ' + err.message;
    return;
  }

  $('setup').style.display = 'none';
  document.documentElement.requestFullscreen?.().then(() => screen.orientation?.lock?.('landscape')).catch(() => {});
  navigator.wakeLock?.request('screen').catch(() => {});

  state.hintUntil = performance.now() + 8000;
  toast('SYSTEMS ONLINE');
  connect();
  setInterval(sendPose, 50);
  requestAnimationFrame(frame);
};

// ---------------------------------------------------------------------------
// Frames

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);

// Graph frame -> headset world: turn clockwise by the calibrated heading
function fromGraph(v) {
  const c = (cal.yaw ?? 0) * DEG;
  return [v[0] * Math.cos(c) + v[1] * Math.sin(c), -v[0] * Math.sin(c) + v[1] * Math.cos(c), v[2]];
}
function toGraph(v) {
  const c = (cal.yaw ?? 0) * DEG;
  return [v[0] * Math.cos(c) - v[1] * Math.sin(c), v[0] * Math.sin(c) + v[1] * Math.cos(c), v[2]];
}

// ---------------------------------------------------------------------------
// Beacon link

let ws = null;
function connect() {
  ws = new WebSocket(`wss://${location.host}/ws`);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'beacon') state.beacon = msg.online === false ? null : { ...msg, at: performance.now() };
    else if (msg.type === 'bstate') onBeaconState(msg);
    else if (msg.type === 'imu') onImuEvent(msg);
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

function beaconCmd(cmd) {
  if (ws?.readyState === 1) ws.send(JSON.stringify({ type: 'cmd', cmd }));
}

const beaconLive = () => !!(state.beacon && performance.now() - state.beacon.at < 3000);

function onBeaconState(msg) {
  const now = performance.now();
  Object.assign(bcn, { p: msg.p, q: msg.q, still: msg.still, air: msg.air, at: now });
  if (!msg.cal) return;
  bcn.trail.push({ t: now, p: msg.p, air: msg.air });
  while (bcn.trail.length && now - bcn.trail[0].t > TRAIL_SECONDS * 1000) bcn.trail.shift();
  if (msg.air && lastThrow && !lastThrow.airtime) lastThrow.peak = Math.max(lastThrow.peak, msg.p[2]);
}

function onImuEvent(msg) {
  if (msg.event === 'cal') {
    if (msg.result === 'ok' && cal.pendingYaw !== null) {
      cal.yaw = cal.pendingYaw;
      zeroPhone();
      bcn.trail = [];
      toast('CALIBRATED // ORIGIN SET');
    } else if (msg.result === 'moving') {
      toast('MOVED // HOLD STILL, TRY AGAIN');
    } else if (msg.result === 'noimu') {
      toast('BEACON HAS NO IMU');
    }
    cal.pendingYaw = null;
  } else if (msg.event === 'zero') {
    bcn.trail = [];
  } else if (msg.event === 'launch') {
    // The server sends the release position as `d` (VAE1 launch ... vx vy vz px py pz)
    lastThrow = { v: msg.v, p0: msg.d, peak: msg.d[2] };
    toast('LAUNCH');
  } else if (msg.event === 'land') {
    if (!lastThrow || lastThrow.airtime) return;
    Object.assign(lastThrow, { airtime: msg.airtime / 1000, impactG: msg.impactG, spin: msg.spin, p1: bcn.p });
    toast('IMPACT');
  }
}

// The phone's pose for the 3D graph: camera position + its device axes, all in the graph frame
function sendPose() {
  if (ws?.readyState !== 1 || !rot || cal.yaw === null) return;
  const col = (j) => toGraph([rot[j], rot[3 + j], rot[6 + j]]);
  ws.send(JSON.stringify({ type: 'phone', p: phone.p, x: col(0), y: col(1), z: col(2) }));
}

// ---------------------------------------------------------------------------
// Head tracking (orientation) and the phone's own position (dead reckoning)

let rot = null; // device -> world rotation matrix: column j is device axis j in world coordinates
let fwd = null; // smoothed look direction (world)
let upDev = null; // smoothed world-up expressed in device coordinates

window.addEventListener('deviceorientation', (e) => {
  if (e.alpha === null) return;
  state.gotSensor = true;
  const a = e.alpha * DEG, b = e.beta * DEG, g = e.gamma * DEG;
  const cA = Math.cos(a), sA = Math.sin(a), cB = Math.cos(b), sB = Math.sin(b), cG = Math.cos(g), sG = Math.sin(g);
  // W3C Z-X'-Y'' intrinsic rotation
  rot = [
    cA * cG - sA * sB * sG, -cB * sA, cG * sA * sB + cA * sG,
    cG * sA + cA * sB * sG, cA * cB, sA * sG - cA * cG * sB,
    -cB * sG, sB, cB * cG,
  ];
});

// Same tricks as the beacon: zero-velocity updates whenever the phone is still, and a speed cap
const PHONE_QUIET_S = 0.25;
window.addEventListener('devicemotion', (e) => {
  const a = e.acceleration; // gravity already removed by the OS, device axes, m/s²
  if (!a || a.x === null || !rot || cal.yaw === null) return;
  let dt = e.interval || 1 / 60;
  if (dt > 1) dt /= 1000; // the spec says ms, older Safari reports seconds
  dt = Math.min(dt, 0.1);
  const world = [
    rot[0] * a.x + rot[1] * a.y + rot[2] * a.z,
    rot[3] * a.x + rot[4] * a.y + rot[5] * a.z,
    rot[6] * a.x + rot[7] * a.y + rot[8] * a.z,
  ];
  const acc = toGraph(world);
  const r = e.rotationRate;
  const spin = r ? Math.hypot(r.alpha || 0, r.beta || 0, r.gamma || 0) : 0;
  phone.quietFor = len(acc) < 0.3 && spin < 20 ? phone.quietFor + dt : 0;
  for (let i = 0; i < 3; i++) phone.v[i] += acc[i] * dt;
  if (phone.quietFor > PHONE_QUIET_S) phone.v = [0, 0, 0];
  const speed = len(phone.v);
  if (speed > 5) phone.v = phone.v.map((x) => (x * 5) / speed);
  for (let i = 0; i < 3; i++) phone.p[i] += phone.v[i] * dt;
});

function zeroPhone() {
  phone.p = [0, 0, 0];
  phone.v = [0, 0, 0];
  phone.trail = [];
}

const lerpVec = (a, b, t) => {
  if (!a) return b;
  const v = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const n = Math.hypot(...v) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
};

const wrap180 = (d) => ((((d + 180) % 360) + 360) % 360) - 180;
const wrap360 = (d) => ((d % 360) + 360) % 360;

function updatePose(now) {
  if (state.gotSensor && rot) {
    // Rear camera looks out the back of the phone: device -Z
    fwd = lerpVec(fwd, [-rot[2], -rot[5], -rot[8]], 0.5);
    upDev = lerpVec(upDev, [rot[6], rot[7], rot[8]], 0.5);
    state.yaw = Math.atan2(fwd[0], fwd[1]) / DEG;
    state.pitch = Math.asin(Math.max(-1, Math.min(1, fwd[2]))) / DEG;

    // Roll: where world-up points on the (possibly rotated) screen
    const sa = (screen.orientation?.angle ?? 0) * DEG;
    const sx = upDev[0] * Math.cos(sa) - upDev[1] * Math.sin(sa);
    const sy = upDev[0] * Math.sin(sa) + upDev[1] * Math.cos(sa);
    state.roll = Math.atan2(sx, sy) / DEG;
  }
  const last = phone.trail[phone.trail.length - 1];
  if (!last || now - last.t > 50) phone.trail.push({ t: now, p: [...phone.p] });
  while (phone.trail.length && now - phone.trail[0].t > TRAIL_SECONDS * 1000) phone.trail.shift();
}

// ---------------------------------------------------------------------------
// Calibration: phone lying face up on the beacon, top edge along the IMU's Y arrow

function calibrate() {
  if (!beaconLive()) return toast('NO BEACON LINK');
  if (state.gotSensor && rot) {
    if (Math.abs(rot[8]) < 0.9) return toast('LAY THE PHONE FLAT ON THE BEACON');
    cal.pendingYaw = Math.atan2(rot[1], rot[4]) / DEG; // heading of the phone's top edge (device +Y)
  } else {
    // Laptop: the webcam looks at you, so a beacon whose Y arrow points at the screen faces heading 180
    cal.pendingYaw = 180;
  }
  beaconCmd('cal');
  toast('CALIBRATING // HOLD STILL');
}

// Hands-free: lying face up and still for AUTO_CAL_MS (with the beacon still too) calibrates by itself,
// once per placement. Picking the phone up re-arms it.
const AUTO_CAL_MS = 2000;
const flat = { since: 0, heading: 0, done: false };

function autoCalibrate(now) {
  if (!state.gotSensor || !rot) return;
  if (rot[8] < 0.95) { // not lying face up
    flat.since = 0;
    flat.done = false;
    return;
  }
  const heading = Math.atan2(rot[1], rot[4]) / DEG;
  const bc = state.beacon;
  // Before calibration the beacon's spin still includes its gyro bias (~5 °/s), hence the loose limit
  const beaconStill = beaconLive() && bc.imu && bc.spin < 12 && !bc.airborne;
  if (!flat.since || Math.abs(wrap180(heading - flat.heading)) > 2 || !beaconStill) {
    flat.since = now;
    flat.heading = heading;
    return;
  }
  if (!flat.done && now - flat.since > AUTO_CAL_MS) {
    flat.done = true;
    calibrate();
  }
}

// Beacon touching the phone: put both back at the same point, which wipes out the drift
function rezero() {
  if (cal.yaw === null) return toast('CALIBRATE FIRST');
  beaconCmd('zero');
  zeroPhone();
  bcn.trail = [];
  toast('RE-ZEROED');
}

// ---------------------------------------------------------------------------
// Input: tap = clear trail, hold = re-zero

let down = null;
window.addEventListener('pointerdown', (e) => {
  if ($('setup').style.display !== 'none') return;
  down = { x: e.clientX, y: e.clientY, t: performance.now() };
});
window.addEventListener('pointerup', (e) => {
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y) > 10;
  if (!moved) performance.now() - down.t > 600 ? rezero() : clearTrail();
  down = null;
});
window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k === 'c') clearTrail();
  if (k === 'k') calibrate();
  if (k === 'z') rezero();
});

function toast(text) {
  state.toast = text;
  state.toastUntil = performance.now() + 1800;
}

function clearTrail() {
  bcn.trail = [];
  lastThrow = null;
  toast('TRAIL CLEARED');
}

// ---------------------------------------------------------------------------
// Camera geometry

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.round(window.innerWidth * dpr);
  const H = Math.round(window.innerHeight * dpr);
  if (screenCanvas.width !== W || screenCanvas.height !== H) {
    screenCanvas.width = W;
    screenCanvas.height = H;
  }
  const ew = settings.stereo ? Math.round(W / 2) : W;
  if (eye.width !== ew || eye.height !== H) {
    eye.width = ew;
    eye.height = H;
  }
  return dpr;
}

// Map the video onto the eye with "cover" scaling; returns the transform.
function videoTransform(w, h) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(w / vw, h / vh);
  return { s, ox: (w - vw * s) / 2, oy: (h - vh * s) / 2, vw, vh };
}

// Pixels-per-unit-tangent of the passthrough as displayed, so drawings line up with the real room.
function focalPx(w, h) {
  if (!video.videoWidth) return w / 2 / Math.tan((settings.passFov / 2) * DEG);
  const t = videoTransform(w, h);
  return (t.vw * t.s) / 2 / Math.tan((settings.passFov / 2) * DEG);
}

function dirVec(yawDeg, pitchDeg) {
  const y = yawDeg * DEG, p = pitchDeg * DEG;
  return [Math.sin(y) * Math.cos(p), Math.cos(y) * Math.cos(p), Math.sin(p)];
}

// Forward / right / up unit vectors for a view at the given heading and pitch
function basis(headingDeg, pitchDeg) {
  const H = headingDeg * DEG, P = pitchDeg * DEG;
  return {
    F: dirVec(headingDeg, pitchDeg),
    R: [Math.cos(H), -Math.sin(H), 0],
    U: [-Math.sin(H) * Math.sin(P), -Math.cos(H) * Math.sin(P), Math.cos(P)],
  };
}

// The camera for this frame. A graph-frame point -> un-rolled screen offset from center
// (draw inside a roll-rotated context).
let cam = null;
function setupCamera(w, h) {
  cam = { f: focalPx(w, h), view: basis(state.yaw, state.pitch), eye: phone.p };
}
function project(g) {
  const p = fromGraph(sub(g, cam.eye));
  const z = dot(p, cam.view.F);
  return { x: (cam.f * dot(p, cam.view.R)) / z, y: (-cam.f * dot(p, cam.view.U)) / z, z };
}

// ---------------------------------------------------------------------------
// Rendering

const NEAR = 0.1; // m: don't draw anything closer than this in front of the camera

function stroke3(ctx, pts) {
  ctx.beginPath();
  let pen = false;
  for (const p of pts) {
    const q = project(p);
    if (q.z < NEAR) { pen = false; continue; }
    pen ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y);
    pen = true;
  }
  ctx.stroke();
}

// A line in 3D drawn as a tube: each segment is as wide as `radius` metres would look at its distance,
// so near parts are fat and far parts thin, which is what makes it read as an object in the room.
function tube3(ctx, pts, radius, minPx = 1.5) {
  let prev = null;
  for (const p of pts) {
    const q = project(p);
    if (q.z < NEAR) { prev = null; continue; }
    if (prev) {
      ctx.lineWidth = Math.max(minPx, Math.min(60, (cam.f * radius * 2) / ((q.z + prev.z) / 2)));
      ctx.beginPath();
      ctx.moveTo(prev.x, prev.y);
      ctx.lineTo(q.x, q.y);
      ctx.stroke();
    }
    prev = q;
  }
}

function label3(ctx, p, text, dy = 0) {
  const q = project(p);
  if (q.z < NEAR) return;
  ctx.fillText(text, q.x, q.y + dy);
}

const fmt = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);
const coord = (p) => `X${fmt(p[0])} Y${fmt(p[1])} Z${fmt(p[2])}`;

// Everything that lives in the room, drawn in a roll-rotated context
function drawWorld(ctx, w, h, now) {
  const fs = Math.round(h / 34);
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.rotate(state.roll * DEG);
  ctx.lineCap = ctx.lineJoin = 'round';
  ctx.font = `${fs}px Menlo, monospace`;
  ctx.textAlign = 'center';

  // Ground grid on the calibration plane (Z = 0), 2 m around the origin
  ctx.strokeStyle = 'rgba(57,255,106,0.18)';
  ctx.lineWidth = 1;
  for (let i = -2; i <= 2; i += 0.5) {
    stroke3(ctx, [[i, -2, 0], [i, 2, 0]]);
    stroke3(ctx, [[-2, i, 0], [2, i, 0]]);
  }

  // Origin axes at the calibration spot: X right, Y along the arrow, Z up
  ctx.lineWidth = Math.max(2, w / 400);
  for (const [v, name, color] of [[[1, 0, 0], 'X', '#ff6b6b'], [[0, 1, 0], 'Y', HUD], [[0, 0, 1], 'Z', '#6bb8ff']]) {
    ctx.strokeStyle = ctx.fillStyle = color;
    stroke3(ctx, [[0, 0, 0], v.map((x) => x * 0.3)]);
    label3(ctx, v.map((x) => x * 0.36), name);
  }

  if (bcn.p && bcn.trail.length) {
    // Trail: fades with age, yellow where it was flying
    const pts = bcn.trail;
    for (let i = 1; i < pts.length; i++) {
      const age = (now - pts[i].t) / (TRAIL_SECONDS * 1000);
      ctx.globalAlpha = Math.max(0.1, 1 - age);
      ctx.strokeStyle = pts[i].air ? PATH : HUD;
      tube3(ctx, [pts[i - 1].p, pts[i].p], 0.008, 2);
    }
    ctx.globalAlpha = 1;

    // Shadow line down to the Z = 0 plane: shows height
    const p = bcn.p;
    ctx.strokeStyle = HUD_DIM;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 6]);
    stroke3(ctx, [p, [p[0], p[1], 0]]);
    ctx.setLineDash([]);
    stroke3(ctx, Array.from({ length: 25 }, (_, i) => {
      const a = (i / 24) * Math.PI * 2;
      return [p[0] + 0.06 * Math.cos(a), p[1] + 0.06 * Math.sin(a), 0];
    }));

    // The beacon itself
    const q = project(p);
    if (q.z > NEAR) {
      const r = Math.max(5, (cam.f * 0.03) / q.z);
      ctx.fillStyle = bcn.air ? PATH : '#fff';
      ctx.shadowColor = bcn.air ? PATH : HUD;
      ctx.shadowBlur = 18;
      ctx.beginPath();
      ctx.arc(q.x, q.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = HUD;
      ctx.fillText(`BCN ${coord(p)}`, q.x, q.y - r - fs * 0.6);
      ctx.fillText(`${len(sub(p, phone.p)).toFixed(2)} m`, q.x, q.y + r + fs * 0.9);
    }
  }
  ctx.restore();
}

// When the beacon is out of view, an arrow at the edge of the screen points to it
function drawGuide(ctx, w, h) {
  if (!bcn.p) return;
  const q = project(bcn.p);
  if (q.z > NEAR && Math.abs(q.x) < w * 0.42 && Math.abs(q.y) < h * 0.42) return;
  // Direction on screen, valid even when the beacon is behind you
  const rel = fromGraph(sub(bcn.p, phone.p));
  let dx = dot(rel, cam.view.R), dy = -dot(rel, cam.view.U);
  const n = Math.hypot(dx, dy) || 1;
  dx /= n; dy /= n;
  const r = Math.min(w, h) * 0.38, s = h / 28;
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.rotate(state.roll * DEG);
  ctx.translate(dx * r, dy * r);
  ctx.rotate(Math.atan2(dy, dx));
  ctx.fillStyle = PATH;
  ctx.shadowColor = 'rgba(0,0,0,0.9)';
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.moveTo(s, 0); ctx.lineTo(-s * 0.6, -s * 0.7); ctx.lineTo(-s * 0.6, s * 0.7);
  ctx.closePath();
  ctx.fill();
  ctx.rotate(-Math.atan2(dy, dx));
  ctx.font = `${Math.round(h / 40)}px Menlo, monospace`;
  ctx.textAlign = 'center';
  ctx.fillText('BCN', -dx * s * 1.8, -dy * s * 1.8 + 4);
  ctx.restore();
}

function drawHUD(ctx, w, h, now) {
  const cx = w / 2, cy = h / 2;
  const ppd = focalPx(w, h) * DEG; // pixels per degree near the center of view
  const fs = Math.round(h / 30);
  const small = Math.round(fs * 0.75);

  ctx.save();
  ctx.strokeStyle = HUD;
  ctx.fillStyle = HUD;
  ctx.lineWidth = Math.max(1.5, w / 450);
  ctx.font = `${fs}px Menlo, monospace`;
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0,0,0,0.9)';
  ctx.shadowBlur = 4;

  // Pitch ladder, rotated with head roll so it stays locked to the real horizon
  ctx.save();
  ctx.beginPath();
  ctx.rect(cx - w * 0.3, cy - h * 0.32, w * 0.6, h * 0.64);
  ctx.clip();
  ctx.translate(cx, cy);
  ctx.rotate(state.roll * DEG);
  ctx.globalAlpha = 0.4;
  for (let p = -90; p <= 90; p += 10) {
    const y = -(p - state.pitch) * ppd;
    if (Math.abs(y) > h * 0.45) continue;
    const half = p === 0 ? w * 0.28 : w * 0.09;
    const gap = w * 0.05;
    ctx.setLineDash(p < 0 ? [w / 80, w / 120] : []);
    ctx.beginPath();
    ctx.moveTo(-half, y); ctx.lineTo(-gap, y);
    ctx.moveTo(gap, y); ctx.lineTo(half, y);
    ctx.stroke();
    if (p !== 0) {
      ctx.textAlign = 'right'; ctx.fillText(Math.abs(p), -half - fs * 0.3, y);
      ctx.textAlign = 'left'; ctx.fillText(Math.abs(p), half + fs * 0.3, y);
    }
  }
  ctx.setLineDash([]);
  ctx.restore();

  // Boresight / waterline symbol
  const b = w * 0.025;
  ctx.beginPath();
  ctx.moveTo(cx - b * 2, cy); ctx.lineTo(cx - b, cy); ctx.lineTo(cx - b / 2, cy + b / 2);
  ctx.lineTo(cx, cy); ctx.lineTo(cx + b / 2, cy + b / 2); ctx.lineTo(cx + b, cy); ctx.lineTo(cx + b * 2, cy);
  ctx.stroke();

  // Heading tape
  const tapeY = h * 0.09, tapeW = w * 0.55, span = 40; // degrees across the tape
  const tps = tapeW / span;
  ctx.save();
  ctx.beginPath();
  ctx.rect(cx - tapeW / 2, tapeY - fs * 2, tapeW, fs * 2.5);
  ctx.clip();
  ctx.textAlign = 'center';
  const start = Math.ceil((state.yaw - span / 2) / 5) * 5;
  for (let d = start; d <= state.yaw + span / 2; d += 5) {
    const x = cx + (d - state.yaw) * tps;
    const major = wrap360(d) % 10 === 0;
    ctx.beginPath();
    ctx.moveTo(x, tapeY); ctx.lineTo(x, tapeY - (major ? fs * 0.7 : fs * 0.35));
    ctx.stroke();
    if (major) ctx.fillText(String(wrap360(d)).padStart(3, '0'), x, tapeY - fs * 1.3);
  }
  ctx.restore();
  ctx.textAlign = 'center';
  ctx.strokeRect(cx - fs * 1.6, tapeY + fs * 0.2, fs * 3.2, fs * 1.3);
  ctx.fillText(String(Math.round(wrap360(state.yaw))).padStart(3, '0'), cx, tapeY + fs * 0.85);

  // Side data blocks: status on the left with the beacon's coordinates, the camera's on the right
  const live = beaconLive(), bc = state.beacon;
  const phase = !live ? 'NO LINK' : cal.yaw === null || !bc.cal ? 'UNCAL' : bcn.air ? 'FLIGHT' : bcn.still ? 'STILL' : 'MOVING';
  const lx = cx - w * 0.36, rx = cx + w * 0.36;
  ctx.textAlign = 'left';
  ctx.fillStyle = phase === 'FLIGHT' ? PATH : phase === 'NO LINK' || phase === 'UNCAL' ? '#ff5a3c' : HUD;
  ctx.fillText(phase, lx, cy - fs * 2);
  ctx.fillStyle = HUD;
  ctx.fillText(!live ? 'LINK --' : bc.link === 'usb' ? 'LINK USB' : `LINK ${bc.rssi}`, lx, cy);
  ctx.font = `${small}px Menlo, monospace`;
  if (bcn.p) {
    ctx.fillText('BCN', lx, cy + fs * 1.8);
    bcn.p.forEach((v, i) => ctx.fillText(`${'XYZ'[i]}${fmt(v)}`, lx, cy + fs * (2.6 + i * 0.8)));
  }
  ctx.textAlign = 'right';
  ctx.fillText('CAM', rx, cy + fs * 1.8);
  phone.p.forEach((v, i) => ctx.fillText(`${'XYZ'[i]}${fmt(v)}`, rx, cy + fs * (2.6 + i * 0.8)));
  ctx.font = `${fs}px Menlo, monospace`;
  ctx.fillText(`P ${state.pitch >= 0 ? '+' : ''}${Math.round(state.pitch)}`, rx, cy - fs * 2);
  ctx.fillText(`${state.fps} FPS`, rx, cy);

  // Last throw
  const t = lastThrow;
  if (t) {
    ctx.textAlign = 'center';
    ctx.fillStyle = PATH;
    ctx.font = `${small}px Menlo, monospace`;
    const horiz = Math.hypot(t.v[0], t.v[1]);
    const parts = [`V0 ${len(t.v).toFixed(1)}m/s @ ${Math.round(Math.atan2(t.v[2], horiz) / DEG)}°`, `PEAK Z${fmt(t.peak)}`];
    if (t.airtime) {
      parts.push(`AIR ${t.airtime.toFixed(2)}s`, `DIST ${Math.hypot(t.p1[0] - t.p0[0], t.p1[1] - t.p0[1]).toFixed(2)}m`);
      parts.push(`HIT ${t.impactG.toFixed(1)}g`);
    } else {
      parts.push('IN FLIGHT');
    }
    ctx.fillText(parts.join('   '), cx, h * 0.82);
    ctx.fillStyle = HUD;
  }

  // Corner brackets
  const m = w * 0.06, L = w * 0.06;
  ctx.beginPath();
  for (const [x, y, sx, sy] of [[m, m, 1, 1], [w - m, m, -1, 1], [m, h - m, 1, -1], [w - m, h - m, -1, -1]]) {
    ctx.moveTo(x, y + sy * L); ctx.lineTo(x, y); ctx.lineTo(x + sx * L, y);
  }
  ctx.stroke();

  // Toast + guidance
  ctx.textAlign = 'center';
  if (now < state.toastUntil) {
    ctx.font = `bold ${Math.round(fs * 1.3)}px Menlo, monospace`;
    ctx.fillText(state.toast, cx, h * 0.68);
  }
  ctx.font = `${small}px Menlo, monospace`;
  if (flat.since && !flat.done) {
    // Counting down to an automatic calibration
    const left = Math.max(1, Math.ceil((AUTO_CAL_MS - (now - flat.since)) / 1000));
    ctx.font = `bold ${Math.round(fs * 1.6)}px Menlo, monospace`;
    ctx.fillText(`CALIBRATING IN ${left}`, cx, h * 0.6);
    ctx.font = `${small}px Menlo, monospace`;
    ctx.fillText('KEEP PHONE AND BEACON STILL', cx, h * 0.93);
  } else if (phase === 'UNCAL') {
    ctx.fillText('LAY PHONE FACE UP ON BEACON, TOP EDGE ALONG Y ARROW', cx, h * 0.93);
  } else if (phase === 'NO LINK') {
    ctx.fillText('WAITING FOR BEACON', cx, h * 0.93);
  } else if (now < state.hintUntil) {
    ctx.fillText('TAP: CLEAR TRAIL   TOUCH BEACON TO PHONE + HOLD: RE-ZERO', cx, h * 0.93);
  }
  ctx.restore();
}

let frames = 0, fpsT = 0;

function frame(now) {
  requestAnimationFrame(frame);
  const dpr = resize();
  updatePose(now);
  autoCalibrate(now);

  const w = eye.width, h = eye.height;
  eyeCtx.clearRect(0, 0, w, h);
  if (video.videoWidth) {
    const t = videoTransform(w, h);
    eyeCtx.drawImage(video, t.ox, t.oy, t.vw * t.s, t.vh * t.s);
  }
  setupCamera(w, h);
  if (cal.yaw !== null) {
    drawWorld(eyeCtx, w, h, now);
    drawGuide(eyeCtx, w, h);
  }
  drawHUD(eyeCtx, w, h, now);

  // Compose to screen: one copy per eye, shifted for lens spacing
  const W = screenCanvas.width, H = screenCanvas.height;
  screenCtx.fillStyle = '#000';
  screenCtx.fillRect(0, 0, W, H);
  if (settings.stereo) {
    const shift = settings.ipd * dpr;
    for (const [x0, dir] of [[0, 1], [w, -1]]) {
      screenCtx.save();
      screenCtx.beginPath();
      screenCtx.rect(x0, 0, w, H);
      screenCtx.clip();
      screenCtx.drawImage(eye, x0 + dir * shift, 0);
      screenCtx.restore();
    }
    screenCtx.fillStyle = '#222';
    screenCtx.fillRect(w - 1, 0, 2, H);
  } else {
    screenCtx.drawImage(eye, 0, 0);
  }

  frames++;
  if (now - fpsT > 1000) {
    state.fps = frames;
    frames = 0;
    fpsT = now;
  }
}
