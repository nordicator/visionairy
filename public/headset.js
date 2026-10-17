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

const settings = { stereo: true, fov: 90, ipd: 0, passFov: 65 };
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
};

// ---------------------------------------------------------------------------
// Setup screen

for (const id of ['fov', 'ipd', 'passFov']) {
  const saved = localStorage.getItem('hs.' + id);
  if (saved !== null) $(id).value = saved;
}
const savedStereo = localStorage.getItem('hs.stereo');
if (savedStereo !== null) $('stereo').checked = savedStereo === '1';

function readSettings() {
  settings.stereo = $('stereo').checked;
  settings.fov = +$('fov').value;
  settings.ipd = +$('ipd').value;
  settings.passFov = +$('passFov').value;
  $('fovVal').textContent = settings.fov + '°';
  $('ipdVal').textContent = settings.ipd;
  $('passFovVal').textContent = settings.passFov + '°';
  for (const id of ['fov', 'ipd', 'passFov']) localStorage.setItem('hs.' + id, $(id).value);
  localStorage.setItem('hs.stereo', settings.stereo ? '1' : '0');
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
          const cam = state.cameras.get(c.slot) || { bitmap: null, decoding: false };
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
        cam.bitmap?.close();
        cam.bitmap = bmp;
      })
      .catch(() => {})
      .finally(() => (cam.decoding = false));
  };
  ws.onclose = () => setTimeout(connect, 1000);
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
  state.yawOffset = rawYaw;
  state.plateValid = false;
  toast('RECENTERED');
}

// ---------------------------------------------------------------------------
// Input: tap = next mode, hold = recenter, drag = look (laptop)

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

function focalPx(w) {
  return w / 2 / Math.tan((settings.fov / 2) * DEG);
}

function dirVec(yawDeg, pitchDeg) {
  const y = yawDeg * DEG, p = pitchDeg * DEG;
  return [Math.sin(y) * Math.cos(p), Math.cos(y) * Math.cos(p), Math.sin(p)];
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

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
    const D = dirVec(cam.info.heading, cam.info.pitch);
    const z = dot(D, F);
    if (z < 0.2) continue;
    visible.push({ cam, x: (f * dot(D, R)) / z, y: (-f * dot(D, U)) / z, z });
  }
  // Furthest from center first, so the best-aligned camera ends up on top
  visible.sort((a, b) => a.z - b.z);

  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.rotate(state.roll * DEG);
  for (const v of visible) {
    const bmp = v.cam.bitmap;
    const iw = (2 * f * Math.tan((v.cam.info.fov / 2) * DEG)) / v.z;
    const ih = (iw * bmp.height) / bmp.width;
    ctx.globalAlpha = Math.min(1, (v.z - 0.2) * 3);
    ctx.drawImage(bmp, v.x - iw / 2, v.y - ih / 2, iw, ih);
  }
  ctx.restore();
  return visible.length;
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
  let source = 'DAS';
  if (!drawDAS(fxCtx, w, h)) {
    if (state.plateValid) {
      // Shift the plate to compensate for head rotation since it was captured
      const ppd = (t.vw * t.s) / settings.passFov;
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
    const x = cx + wrap180(cam.info.heading - state.yaw) * tps;
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
