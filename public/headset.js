// VisionAiry headset: head tracking + room camera fusion + see-through hands + F-35 style HUD.
//
// Coordinate convention: yaw is degrees clockwise from "straight ahead" (set by recenter),
// pitch is degrees up from the horizon. Room cameras report the same heading/pitch for their lens.

import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

const HUD = '#39ff6a';
const HUD_DIM = 'rgba(57,255,106,0.45)';
const MODES = ['XRAY', 'DAS', 'PASS'];
const DEG = Math.PI / 180;

const $ = (id) => document.getElementById(id);
const screenCanvas = $('screen');
const screenCtx = screenCanvas.getContext('2d');

// Offscreen layers: one eye is rendered once, then copied into both halves of the screen.
const eye = document.createElement('canvas');
const eyeCtx = eye.getContext('2d');
const fx = document.createElement('canvas'); // see-through fill, cut to the hand shape
const fxCtx = fx.getContext('2d');
const mask = document.createElement('canvas'); // soft hand silhouette
const maskCtx = mask.getContext('2d');
const plate = document.createElement('canvas'); // last hand-free passthrough frame
const plateCtx = plate.getContext('2d');

const video = document.createElement('video');
video.playsInline = true;
video.muted = true;

const settings = { stereo: true, fov: 90, ipd: 0, passFov: 65, autoSync: true };
const state = {
  mode: 0,
  yaw: 0, pitch: 0, roll: 0,
  yawOffset: 0,
  gotSensor: false,
  manualYaw: 0, manualPitch: 0,
  cameras: new Map(), // slot -> { info, bitmap, decoding }
  hands: [],
  plateYaw: 0, platePitch: 0, plateValid: false,
  toast: '', toastUntil: 0,
  hintUntil: 0,
  fps: 0,
  syncScore: 0, syncAt: 0,
};
window.visionairy = { state, settings }; // poke at it from the dev console

// ---------------------------------------------------------------------------
// Setup screen

for (const id of ['fov', 'ipd', 'passFov']) {
  const saved = localStorage.getItem('hs.' + id);
  if (saved !== null) $(id).value = saved;
}
for (const id of ['stereo', 'autoSync']) {
  const saved = localStorage.getItem('hs.' + id);
  if (saved !== null) $(id).checked = saved === '1';
}

function readSettings() {
  settings.stereo = $('stereo').checked;
  settings.autoSync = $('autoSync').checked;
  settings.fov = +$('fov').value;
  settings.ipd = +$('ipd').value;
  settings.passFov = +$('passFov').value;
  $('fovVal').textContent = settings.fov + '°';
  $('ipdVal').textContent = settings.ipd;
  $('passFovVal').textContent = settings.passFov + '°';
  for (const id of ['fov', 'ipd', 'passFov']) localStorage.setItem('hs.' + id, $(id).value);
  localStorage.setItem('hs.stereo', settings.stereo ? '1' : '0');
  localStorage.setItem('hs.autoSync', settings.autoSync ? '1' : '0');
}
document.querySelectorAll('#setup input').forEach((el) => el.addEventListener('input', readSettings));
readSettings();

$('go').onclick = async () => {
  $('err').textContent = '';
  try {
    // iOS gates motion sensors behind an explicit permission prompt from a user gesture
    if (typeof DeviceOrientationEvent !== 'undefined' && DeviceOrientationEvent.requestPermission) {
      const res = await DeviceOrientationEvent.requestPermission();
      if (res !== 'granted') throw new Error('motion sensor permission denied');
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

  state.hintUntil = performance.now() + 6000;
  toast('SYSTEMS ONLINE');
  connect();
  loadHandModel();
  requestAnimationFrame(frame);
};

// ---------------------------------------------------------------------------
// Room camera feeds

function connect() {
  const ws = new WebSocket(`wss://${location.host}/ws?role=viewer`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => {
    if (typeof e.data === 'string') {
      const msg = JSON.parse(e.data);
      if (msg.type === 'cameras') {
        const live = new Set(msg.list.map((c) => c.slot));
        for (const slot of state.cameras.keys()) if (!live.has(slot)) state.cameras.delete(slot);
        for (const c of msg.list) {
          const cam = state.cameras.get(c.slot) || { bitmap: null, raw: null, decoding: false, trim: { yaw: 0, pitch: 0 } };
          cam.info = c;
          state.cameras.set(c.slot, cam);
        }
      }
      return;
    }
    const slot = new Uint8Array(e.data, 0, 1)[0];
    const cam = state.cameras.get(slot);
    if (!cam || cam.decoding) return; // still decoding the previous frame: drop this one
    cam.decoding = true;
    createImageBitmap(new Blob([new Uint8Array(e.data, 1)], { type: 'image/jpeg' }))
      .then((bmp) => {
        cam.bitmap = feather(bmp, cam.bitmap);
        cam.raw?.close();
        cam.raw = bmp; // unfeathered copy, used by visual sync
      })
      .catch(() => {})
      .finally(() => (cam.decoding = false));
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

// Fade each frame's edges to transparent so overlapping cameras blend instead of cutting hard.
const featherMasks = new Map(); // "w x h" -> mask canvas

function featherMask(w, h) {
  const key = w + 'x' + h;
  if (featherMasks.has(key)) return featherMasks.get(key);
  const m = document.createElement('canvas');
  m.width = w;
  m.height = h;
  const c = m.getContext('2d');
  const img = c.createImageData(w, h);
  const edge = w * 0.12;
  const ramp = (d) => Math.min(1, d / edge);
  for (let y = 0; y < h; y++) {
    const ay = ramp(Math.min(y, h - 1 - y));
    for (let x = 0; x < w; x++) img.data[(y * w + x) * 4 + 3] = 255 * ay * ramp(Math.min(x, w - 1 - x));
  }
  c.putImageData(img, 0, 0);
  featherMasks.set(key, m);
  return m;
}

function feather(bmp, reuse) {
  const out = reuse || document.createElement('canvas');
  if (out.width !== bmp.width || out.height !== bmp.height) {
    out.width = bmp.width;
    out.height = bmp.height;
  }
  const c = out.getContext('2d');
  c.globalCompositeOperation = 'copy';
  c.drawImage(bmp, 0, 0);
  c.globalCompositeOperation = 'destination-in';
  c.drawImage(featherMask(bmp.width, bmp.height), 0, 0);
  return out;
}

// ---------------------------------------------------------------------------
// Hand tracking

let handLandmarker = null;
let lastHandRun = 0;

async function loadHandModel() {
  try {
    const files = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    const opts = (delegate) => ({
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate,
      },
      runningMode: 'VIDEO',
      numHands: 2,
    });
    try {
      handLandmarker = await HandLandmarker.createFromOptions(files, opts('GPU'));
    } catch {
      handLandmarker = await HandLandmarker.createFromOptions(files, opts('CPU'));
    }
    toast('HAND TRACKING ONLINE');
  } catch (err) {
    console.error(err);
    toast('HAND TRACKING OFFLINE');
  }
}

function updateHands(now) {
  if (!handLandmarker || MODES[state.mode] !== 'XRAY' || now - lastHandRun < 40 || !video.videoWidth) return;
  lastHandRun = now;
  try {
    state.hands = handLandmarker.detectForVideo(video, now).landmarks || [];
  } catch (err) {
    console.error(err);
    state.hands = [];
  }
}

const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const PALM = [0, 1, 5, 9, 13, 17];

// ---------------------------------------------------------------------------
// Head tracking

let rot = null; // device -> world rotation matrix from the latest sensor event
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

const lerpVec = (a, b, t) => {
  if (!a) return b;
  const v = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const n = Math.hypot(...v) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
};

const wrap180 = (d) => ((((d + 180) % 360) + 360) % 360) - 180;
const wrap360 = (d) => ((d % 360) + 360) % 360;

let rawYaw = 0;

function updatePose() {
  if (state.gotSensor && rot) {
    // Rear camera looks out the back of the phone: device -Z
    fwd = lerpVec(fwd, [-rot[2], -rot[5], -rot[8]], 0.5);
    upDev = lerpVec(upDev, [rot[6], rot[7], rot[8]], 0.5);
    rawYaw = Math.atan2(fwd[0], fwd[1]) / DEG;
    state.pitch = Math.asin(Math.max(-1, Math.min(1, fwd[2]))) / DEG;

    // Roll: where world-up points on the (possibly rotated) screen
    const sa = (screen.orientation?.angle ?? window.orientation ?? 0) * DEG;
    const sx = upDev[0] * Math.cos(sa) - upDev[1] * Math.sin(sa);
    const sy = upDev[0] * Math.sin(sa) + upDev[1] * Math.cos(sa);
    state.roll = Math.atan2(sx, sy) / DEG;
  } else {
    rawYaw = state.manualYaw;
    state.pitch = state.manualPitch;
    state.roll = 0;
  }
  state.yaw = wrap180(rawYaw - state.yawOffset);
}

function recenter() {
  state.plateValid = false;
  const hit = syncGlobal();
  if (hit) {
    toast('SYNCED TO ' + hit.name);
  } else {
    // No camera matched what we see: fall back to "I'm facing heading 0"
    state.yawOffset = rawYaw;
    toast('RECENTERED (NO MATCH)');
  }
}

// ---------------------------------------------------------------------------
// Input: tap = next mode, hold = sync/recenter, drag = look (laptop)

let down = null;
window.addEventListener('pointerdown', (e) => {
  if ($('setup').style.display !== 'none') return;
  down = { x: e.clientX, y: e.clientY, t: performance.now(), moved: false };
});
window.addEventListener('pointermove', (e) => {
  if (!down) return;
  const dx = e.clientX - down.x, dy = e.clientY - down.y;
  if (Math.hypot(dx, dy) > 10) down.moved = true;
  if (!state.gotSensor && down.moved) {
    state.manualYaw = wrap180(state.manualYaw - e.movementX * 0.25);
    state.manualPitch = Math.max(-89, Math.min(89, state.manualPitch + e.movementY * 0.25));
  }
});
window.addEventListener('pointerup', () => {
  if (!down) return;
  const held = performance.now() - down.t;
  if (!down.moved) held > 600 ? recenter() : setMode(state.mode + 1);
  down = null;
});
window.addEventListener('keydown', (e) => {
  if (e.key >= '1' && e.key <= '3') setMode(+e.key - 1);
  if (e.key === 'r' || e.key === 'R') recenter();
});

function setMode(m) {
  state.mode = (m + MODES.length) % MODES.length;
  state.hands = [];
  toast('MODE ' + MODES[state.mode]);
}

function toast(text) {
  state.toast = text;
  state.toastUntil = performance.now() + 1500;
}

// ---------------------------------------------------------------------------
// Rendering

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.round(window.innerWidth * dpr);
  const H = Math.round(window.innerHeight * dpr);
  if (screenCanvas.width !== W || screenCanvas.height !== H) {
    screenCanvas.width = W;
    screenCanvas.height = H;
  }
  const ew = settings.stereo ? Math.round(W / 2) : W;
  for (const c of [eye, fx, mask]) {
    if (c.width !== ew || c.height !== H) {
      c.width = ew;
      c.height = H;
    }
  }
  return dpr;
}

// Map the headset video onto the eye with "cover" scaling; returns the transform.
function videoTransform(w, h) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(w / vw, h / vh);
  return { s, ox: (w - vw * s) / 2, oy: (h - vh * s) / 2, vw, vh };
}

// Pixels-per-unit-tangent for the current view. In DAS it's the virtual lens (settings.fov). With passthrough
// on screen it must match the phone camera as displayed, or room-camera fills and the HUD won't line up.
function focalPx(w) {
  if (MODES[state.mode] !== 'DAS' && video.videoWidth) {
    const t = videoTransform(w, eye.height);
    return (t.vw * t.s) / 2 / Math.tan((settings.passFov / 2) * DEG);
  }
  return w / 2 / Math.tan((settings.fov / 2) * DEG);
}

function dirVec(yawDeg, pitchDeg) {
  const y = yawDeg * DEG, p = pitchDeg * DEG;
  return [Math.sin(y) * Math.cos(p), Math.cos(y) * Math.cos(p), Math.sin(p)];
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// Forward / right / up unit vectors for a view at the given heading and pitch
function basis(headingDeg, pitchDeg) {
  const H = headingDeg * DEG, P = pitchDeg * DEG;
  return {
    F: dirVec(headingDeg, pitchDeg),
    R: [Math.cos(H), -Math.sin(H), 0],
    U: [-Math.sin(H) * Math.sin(P), -Math.cos(H) * Math.sin(P), Math.cos(P)],
  };
}

// Where a room camera actually points: its configured aim plus what visual sync has learned
function camAim(cam) {
  return [cam.info.heading + cam.trim.yaw, cam.info.pitch + cam.trim.pitch];
}

// ---------------------------------------------------------------------------
// Visual sync. The gyro only knows relative yaw (and drifts), and camera headings are only as good as
// where someone put the camera. So compare edges in the passthrough with each room camera's frame and
// solve for the angle where they overlap best.

const SYNC_W = 96; // analysis width of the passthrough, px
const SYNC_MIN = 0.3; // minimum edge correlation to trust a match
const syncCanvas = document.createElement('canvas');
const syncCtx = syncCanvas.getContext('2d', { willReadFrequently: true });

// Draw something into a small canvas and return its edge strength (Sobel magnitude) per pixel.
function edges(w, h, draw) {
  syncCanvas.width = w;
  syncCanvas.height = h;
  syncCtx.fillStyle = '#000';
  syncCtx.fillRect(0, 0, w, h);
  draw(syncCtx);
  const px = syncCtx.getImageData(0, 0, w, h).data;
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = px[i * 4] * 0.3 + px[i * 4 + 1] * 0.59 + px[i * 4 + 2] * 0.11;
  const e = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1] - g[i - w - 1] - 2 * g[i - 1] - g[i + w - 1];
      const gy = g[i + w - 1] + 2 * g[i + w] + g[i + w + 1] - g[i - w - 1] - 2 * g[i - w] - g[i - w + 1];
      e[i] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return { w, h, e };
}

// Passthrough, de-rolled so it lines up with level room cameras. Uses the inner 80% to avoid rotation corners.
function passEdges() {
  const w = SYNC_W, h = Math.round((SYNC_W * video.videoHeight) / video.videoWidth);
  const img = edges(w, h, (c) => {
    c.translate(w / 2, h / 2);
    c.rotate(-state.roll * DEG);
    c.drawImage(video, -w / 2, -h / 2, w, h);
    c.setTransform(1, 0, 0, 1, 0, 0);
  });
  img.f = w / 2 / Math.tan((settings.passFov / 2) * DEG); // px per unit tangent
  img.x0 = Math.round(w * 0.1); img.x1 = w - img.x0;
  img.y0 = Math.round(h * 0.1); img.y1 = h - img.y0;
  return img;
}

// Room camera frame resampled to the same px-per-tangent as the passthrough
function camEdges(cam, f) {
  const w = Math.round(2 * f * Math.tan((cam.info.fov / 2) * DEG));
  const h = Math.round((w * cam.raw.height) / cam.raw.width);
  return edges(w, h, (c) => c.drawImage(cam.raw, 0, 0, w, h));
}

// Normalized cross-correlation of A (template) against B over a range of offsets.
// Offset (ox, oy) means A pixel (x, y) sits on B pixel (x + ox, y + oy).
function bestOffset(A, B, oxMin, oxMax, oyMin, oyMax) {
  const area = (A.x1 - A.x0) * (A.y1 - A.y0);
  let best = { score: -1 };
  for (let oy = oyMin; oy <= oyMax; oy++) {
    for (let ox = oxMin; ox <= oxMax; ox++) {
      const xa = Math.max(A.x0, -ox), xb = Math.min(A.x1, B.w - ox);
      const ya = Math.max(A.y0, -oy), yb = Math.min(A.y1, B.h - oy);
      const n = (xb - xa) * (yb - ya);
      if (xb <= xa || yb <= ya || n < area * 0.35) continue;
      let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
      for (let y = ya; y < yb; y++) {
        let ia = y * A.w + xa, ib = (y + oy) * B.w + xa + ox;
        for (let x = xa; x < xb; x++, ia++, ib++) {
          const a = A.e[ia], b = B.e[ib];
          sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
        }
      }
      const den = Math.sqrt((n * saa - sa * sa) * (n * sbb - sb * sb));
      if (!den) continue;
      // Favour bigger overlaps so a tiny sliver can't win by luck
      const score = ((n * sab - sa * sb) / den) * Math.min(1, n / (area * 0.6));
      if (score > best.score) best = { score, ox, oy };
    }
  }
  return best;
}

// Turn a match into the true direction of the headset's view center.
function matchedDirection(A, B, cam, m) {
  const sx = A.w / 2 + m.ox - B.w / 2, sy = A.h / 2 + m.oy - B.h / 2;
  const { F, R, U } = basis(...camAim(cam));
  const d = [0, 1, 2].map((k) => F[k] * A.f + sx * R[k] - sy * U[k]);
  const n = Math.hypot(...d);
  return { yaw: Math.atan2(d[0], d[1]) / DEG, pitch: Math.asin(d[2] / n) / DEG };
}

// Expected offset of A's center inside B, given the current pose estimate
function predictedOffset(A, B, cam) {
  const { F, R, U } = basis(...camAim(cam));
  const V = dirVec(state.yaw, state.pitch);
  const z = dot(V, F);
  return { ox: Math.round(B.w / 2 + (A.f * dot(V, R)) / z - A.w / 2), oy: Math.round(B.h / 2 - (A.f * dot(V, U)) / z - A.h / 2) };
}

const syncable = (cam) => cam.raw && cam.info && Math.abs(cam.info.pitch) <= 60; // floor cams have no useful heading

// Hold gesture: search every camera across its whole width, then set the headset's yaw from the best match.
function syncGlobal() {
  if (!video.videoWidth || Math.abs(state.pitch) > 60) return null;
  const A = passEdges();
  const ppd = A.f * DEG;
  let best = null;
  for (const cam of state.cameras.values()) {
    if (!syncable(cam)) continue;
    const B = camEdges(cam, A.f);
    // Pitch is trustworthy (gravity), so only a narrow vertical band needs searching
    const oy = Math.round(B.h / 2 - A.f * Math.tan((state.pitch - camAim(cam)[1]) * DEG) - A.h / 2);
    const m = bestOffset(A, B, -A.w, B.w, oy - Math.round(10 * ppd), oy + Math.round(10 * ppd));
    if (m.score > (best?.m.score ?? SYNC_MIN)) best = { cam, A, B, m };
  }
  if (!best) return null;
  const dir = matchedDirection(best.A, best.B, best.cam, best.m);
  state.yawOffset = rawYaw - dir.yaw;
  best.cam.trim.pitch += state.pitch - dir.pitch;
  state.syncScore = best.m.score;
  state.syncAt = performance.now();
  return best.cam.info;
}

// Continuous: every so often, refine the trim of the camera we're looking at with a small local search.
// This soaks up gyro drift and per-camera placement error.
let lastAutoSync = 0;

function autoSync(now) {
  if (!settings.autoSync || now - lastAutoSync < 400 || !video.videoWidth || state.hands.length) return;
  if (Math.abs(state.pitch) > 60) return;
  lastAutoSync = now;
  const V = dirVec(state.yaw, state.pitch);
  let cam = null, bestZ = 0.85; // only cameras within ~30° of where we're looking
  for (const c of state.cameras.values()) {
    if (!syncable(c)) continue;
    const z = dot(V, dirVec(...camAim(c)));
    if (z > bestZ) { bestZ = z; cam = c; }
  }
  if (!cam) return;
  const A = passEdges();
  const B = camEdges(cam, A.f);
  const p = predictedOffset(A, B, cam);
  const r = Math.round(12 * A.f * DEG); // covers a camera placed up to ~12° off its preset
  const m = bestOffset(A, B, p.ox - r, p.ox + r, p.oy - r, p.oy + r);
  if (m.score < SYNC_MIN) return;
  const dir = matchedDirection(A, B, cam, m);
  // Ease toward the measured error rather than jumping, so one bad match can't yank the view
  const clamp = (v) => Math.max(-30, Math.min(30, v));
  cam.trim.yaw = clamp(cam.trim.yaw + 0.5 * wrap180(state.yaw - dir.yaw));
  cam.trim.pitch = clamp(cam.trim.pitch + 0.5 * (state.pitch - dir.pitch));
  state.syncScore = m.score;
  state.syncAt = now;
}

// Mesh resolution for projecting each room-camera frame. Each cell is drawn as two affine triangles,
// which tracks the true perspective closely without WebGL.
const MESH_X = 6, MESH_Y = 4;

// Project every room camera into the current view. Returns how many were visible.
function drawDAS(ctx, w, h) {
  const f = focalPx(w);
  const y = state.yaw * DEG, p = state.pitch * DEG;
  const F = dirVec(state.yaw, state.pitch);
  const R = [Math.cos(y), -Math.sin(y), 0];
  const U = [-Math.sin(y) * Math.sin(p), -Math.cos(y) * Math.sin(p), Math.cos(p)];

  const visible = [];
  for (const cam of state.cameras.values()) {
    if (!cam.bitmap || !cam.info) continue;
    const z = dot(dirVec(...camAim(cam)), F);
    if (z < 0.2) continue;
    visible.push({ cam, z });
  }
  // Furthest from center first, so the best-aligned camera ends up on top
  visible.sort((a, b) => a.z - b.z);

  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.rotate(state.roll * DEG);
  for (const v of visible) {
    const { info, bitmap: bmp } = v.cam;
    const { F: cF, R: cR, U: cU } = basis(...camAim(v.cam));
    const tx = Math.tan((info.fov / 2) * DEG), ty = (tx * bmp.height) / bmp.width;

    // Project mesh vertices: image (px) -> world ray -> screen
    const pts = [];
    for (let j = 0; j <= MESH_Y; j++) {
      for (let i = 0; i <= MESH_X; i++) {
        const u = (i / MESH_X) * 2 - 1, vv = 1 - (j / MESH_Y) * 2;
        const d = [0, 1, 2].map((k) => cF[k] + u * tx * cR[k] + vv * ty * cU[k]);
        const z = dot(d, F);
        pts.push({ ix: (i / MESH_X) * bmp.width, iy: (j / MESH_Y) * bmp.height, z,
          x: (f * dot(d, R)) / z, y: (-f * dot(d, U)) / z });
      }
    }
    ctx.globalAlpha = Math.min(1, (v.z - 0.2) * 3);
    const at = (i, j) => pts[j * (MESH_X + 1) + i];
    for (let j = 0; j < MESH_Y; j++) {
      for (let i = 0; i < MESH_X; i++) {
        const a = at(i, j), b = at(i + 1, j), c = at(i, j + 1), d = at(i + 1, j + 1);
        if (a.z < 0.05 || b.z < 0.05 || c.z < 0.05 || d.z < 0.05) continue;
        drawTriangle(ctx, bmp, a, b, c);
        drawTriangle(ctx, bmp, b, d, c);
      }
    }
  }
  ctx.restore();
  return visible;
}

// Draw one triangle of an image with the affine map that sends its image corners to its screen corners.
function drawTriangle(ctx, img, p0, p1, p2) {
  const du1 = p1.ix - p0.ix, dv1 = p1.iy - p0.iy, du2 = p2.ix - p0.ix, dv2 = p2.iy - p0.iy;
  const det = du1 * dv2 - du2 * dv1;
  if (!det) return;
  const dx1 = p1.x - p0.x, dy1 = p1.y - p0.y, dx2 = p2.x - p0.x, dy2 = p2.y - p0.y;
  const a = (dx1 * dv2 - dx2 * dv1) / det, c = (du1 * dx2 - du2 * dx1) / det;
  const b = (dy1 * dv2 - dy2 * dv1) / det, d = (du1 * dy2 - du2 * dy1) / det;
  // Grow the clip ~1px around the centroid so neighbouring triangles don't leave hairline cracks
  const mx = (p0.x + p1.x + p2.x) / 3, my = (p0.y + p1.y + p2.y) / 3;
  const grow = (q) => {
    const dx = q.x - mx, dy = q.y - my, n = Math.hypot(dx, dy) || 1;
    return [q.x + dx / n, q.y + dy / n];
  };
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(...grow(p0));
  ctx.lineTo(...grow(p1));
  ctx.lineTo(...grow(p2));
  ctx.closePath();
  ctx.clip();
  ctx.transform(a, b, c, d, p0.x - a * p0.ix - c * p0.iy, p0.y - b * p0.ix - d * p0.iy);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

function drawNoSignal(ctx, w, h) {
  ctx.fillStyle = '#020a04';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(57,255,106,0.08)';
  ctx.lineWidth = 1;
  const step = h / 12;
  for (let x = (w / 2) % step; x < w; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = (h / 2) % step; y < h; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
}

function handPoints(hand, t) {
  return hand.map((p) => [t.ox + p.x * t.vw * t.s, t.oy + p.y * t.vh * t.s]);
}

function drawXray(w, h, t) {
  // 1. Plain passthrough
  eyeCtx.drawImage(video, t.ox, t.oy, t.vw * t.s, t.vh * t.s);

  const hands = state.hands.map((hd) => handPoints(hd, t));

  // 2. While no hands are visible, keep a clean plate of the scene for fallback fill
  if (!hands.length) {
    if (plate.width !== w || plate.height !== h) { plate.width = w; plate.height = h; }
    plateCtx.drawImage(eye, 0, 0);
    state.plateYaw = state.yaw;
    state.platePitch = state.pitch;
    state.plateValid = true;
    return 0;
  }

  // 3. Build what's "behind" the hands: room cameras first, clean plate if none are in view
  fxCtx.globalCompositeOperation = 'source-over';
  fxCtx.clearRect(0, 0, w, h);
  let source;
  const shown = drawDAS(fxCtx, w, h);
  if (shown.length) {
    source = shown[shown.length - 1].cam.info.name; // the best-aligned camera is drawn on top
  } else {
    if (state.plateValid) {
      // Shift the plate to compensate for head rotation since it was captured
      const ppd = focalPx(w) * DEG;
      const dx = -wrap180(state.yaw - state.plateYaw) * ppd;
      const dy = (state.pitch - state.platePitch) * ppd;
      fxCtx.drawImage(plate, dx, dy);
      source = 'MEM';
    } else {
      fxCtx.fillStyle = 'rgba(0,40,10,0.85)';
      fxCtx.fillRect(0, 0, w, h);
      source = 'NONE';
    }
  }

  // 4. Soft-edged hand silhouette
  maskCtx.clearRect(0, 0, w, h);
  maskCtx.save();
  maskCtx.fillStyle = maskCtx.strokeStyle = '#fff';
  maskCtx.shadowColor = '#fff';
  maskCtx.shadowBlur = w / 40;
  maskCtx.lineCap = maskCtx.lineJoin = 'round';
  for (const pts of hands) {
    const size = Math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1]);
    maskCtx.beginPath();
    PALM.forEach((i, k) => (k ? maskCtx.lineTo(...pts[i]) : maskCtx.moveTo(...pts[i])));
    maskCtx.closePath();
    maskCtx.fill();
    maskCtx.lineWidth = size * 0.45;
    maskCtx.stroke();
    maskCtx.lineWidth = size * 0.28;
    maskCtx.beginPath();
    for (const [a, b] of BONES) { maskCtx.moveTo(...pts[a]); maskCtx.lineTo(...pts[b]); }
    maskCtx.stroke();
  }
  maskCtx.restore();

  fxCtx.globalCompositeOperation = 'destination-in';
  fxCtx.drawImage(mask, 0, 0);
  eyeCtx.drawImage(fx, 0, 0);

  // 5. Wireframe skeleton over the now-invisible hand
  eyeCtx.save();
  eyeCtx.strokeStyle = HUD;
  eyeCtx.fillStyle = HUD;
  eyeCtx.lineWidth = Math.max(1.5, w / 500);
  eyeCtx.shadowColor = HUD;
  eyeCtx.shadowBlur = 6;
  for (const pts of hands) {
    eyeCtx.beginPath();
    for (const [a, b] of BONES) { eyeCtx.moveTo(...pts[a]); eyeCtx.lineTo(...pts[b]); }
    eyeCtx.stroke();
    for (const i of [4, 8, 12, 16, 20]) {
      eyeCtx.beginPath();
      eyeCtx.arc(pts[i][0], pts[i][1], w / 180, 0, Math.PI * 2);
      eyeCtx.fill();
    }
  }
  eyeCtx.font = `${Math.round(h / 34)}px Menlo, monospace`;
  eyeCtx.textAlign = 'center';
  eyeCtx.fillText(`HAND TRANSPARENT // SRC ${source}`, w / 2, h * 0.86);
  eyeCtx.restore();
  return hands.length;
}

function drawHUD(ctx, w, h, now) {
  const cx = w / 2, cy = h / 2;
  const f = focalPx(w);
  const ppd = f * DEG; // pixels per degree near the center of view
  const fs = Math.round(h / 30);

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
  for (let p = -90; p <= 90; p += 10) {
    const y = -(p - state.pitch) * ppd;
    if (Math.abs(y) > h * 0.45) continue;
    const half = p === 0 ? w * 0.28 : w * 0.09;
    const gap = w * 0.05;
    ctx.setLineDash(p < 0 ? [w / 80, w / 120] : []);
    ctx.beginPath();
    ctx.moveTo(-half, y); ctx.lineTo(-gap, y);
    ctx.moveTo(gap, y); ctx.lineTo(half, y);
    if (p !== 0) {
      const tick = p > 0 ? fs * 0.5 : -fs * 0.5;
      ctx.moveTo(-half, y); ctx.lineTo(-half, y + tick);
      ctx.moveTo(half, y); ctx.lineTo(half, y + tick);
    }
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
  ctx.rect(cx - tapeW / 2, tapeY - fs * 2, tapeW, fs * 4.5);
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
  // Camera markers on the tape
  ctx.font = `${Math.round(fs * 0.7)}px Menlo, monospace`;
  for (const cam of state.cameras.values()) {
    if (!cam.info) continue;
    const x = cx + wrap180(camAim(cam)[0] - state.yaw) * tps;
    ctx.fillStyle = cam.bitmap ? HUD : HUD_DIM;
    ctx.beginPath();
    ctx.moveTo(x, tapeY + fs * 0.3); ctx.lineTo(x - fs * 0.35, tapeY + fs * 0.9); ctx.lineTo(x + fs * 0.35, tapeY + fs * 0.9);
    ctx.fill();
    ctx.fillText(cam.info.name.slice(0, 10), x, tapeY + fs * 1.6);
  }
  ctx.restore();
  ctx.fillStyle = HUD;
  ctx.font = `${fs}px Menlo, monospace`;
  ctx.textAlign = 'center';
  ctx.strokeRect(cx - fs * 1.6, tapeY + fs * 0.2, fs * 3.2, fs * 1.3);
  ctx.fillText(String(Math.round(wrap360(state.yaw))).padStart(3, '0'), cx, tapeY + fs * 0.85);

  // Side data blocks
  const live = [...state.cameras.values()].filter((c) => c.bitmap).length;
  const lx = cx - w * 0.36, rx = cx + w * 0.36;
  ctx.textAlign = 'left';
  ctx.fillText(MODES[state.mode], lx, cy - fs * 2);
  ctx.fillText(`DAS ${live}/${state.cameras.size}`, lx, cy);
  ctx.fillText(state.gotSensor ? 'IMU OK' : 'IMU --', lx, cy + fs * 2);
  ctx.fillText(now - state.syncAt < 3000 ? `SYNC ${state.syncScore.toFixed(2)}` : 'SYNC --', lx, cy + fs * 4);
  ctx.textAlign = 'right';
  ctx.fillText(`P ${state.pitch >= 0 ? '+' : ''}${Math.round(state.pitch)}`, rx, cy - fs * 2);
  ctx.fillText(`R ${Math.round(state.roll)}`, rx, cy);
  ctx.fillText(`${state.fps} FPS`, rx, cy + fs * 2);

  // Corner brackets
  const m = w * 0.06, L = w * 0.06;
  ctx.beginPath();
  for (const [x, y, sx, sy] of [[m, m, 1, 1], [w - m, m, -1, 1], [m, h - m, 1, -1], [w - m, h - m, -1, -1]]) {
    ctx.moveTo(x, y + sy * L); ctx.lineTo(x, y); ctx.lineTo(x + sx * L, y);
  }
  ctx.stroke();

  // Toast + controls hint
  ctx.textAlign = 'center';
  if (now < state.toastUntil) {
    ctx.font = `bold ${Math.round(fs * 1.4)}px Menlo, monospace`;
    ctx.fillText(state.toast, cx, h * 0.72);
  }
  if (now < state.hintUntil) {
    ctx.font = `${Math.round(fs * 0.75)}px Menlo, monospace`;
    ctx.fillText('TAP: MODE   HOLD: RECENTER', cx, h * 0.93);
  }
  ctx.restore();
}

let frames = 0, fpsT = 0;

function frame(now) {
  requestAnimationFrame(frame);
  const dpr = resize();
  updatePose();
  updateHands(now);
  autoSync(now);

  const w = eye.width, h = eye.height;
  eyeCtx.globalCompositeOperation = 'source-over';
  eyeCtx.clearRect(0, 0, w, h);

  const mode = MODES[state.mode];
  if (mode === 'DAS') {
    drawNoSignal(eyeCtx, w, h);
    drawDAS(eyeCtx, w, h);
  } else if (video.videoWidth) {
    const t = videoTransform(w, h);
    if (mode === 'XRAY') drawXray(w, h, t);
    else eyeCtx.drawImage(video, t.ox, t.oy, t.vw * t.s, t.vh * t.s);
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
