// VisionAiry server: serves the headset over HTTPS and relays the ESP32 beacon's UDP heartbeat to it.
// HTTPS is required: phones refuse camera + gyro access on plain http.

const https = require('https');
const http = require('http');
const dgram = require('dgram');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');
const { WebSocketServer } = require('ws');

const PORT = process.env.PORT || 8443;
const APP_PORT = 8080; // plain ws:// for the native AR app (it can't trust our self-signed cert)
const BEACON_PORT = 4210;
const PUBLIC = path.join(__dirname, 'public');
const CERT_DIR = path.join(__dirname, '.cert');

function lanIPs() {
  return Object.values(os.networkInterfaces())
    .flat()
    .filter((i) => i && i.family === 'IPv4' && !i.internal)
    .map((i) => i.address);
}

// Self-signed cert, generated once. Phones will show a warning — tap through it.
function loadCert() {
  const key = path.join(CERT_DIR, 'key.pem');
  const cert = path.join(CERT_DIR, 'cert.pem');
  if (!fs.existsSync(key) || !fs.existsSync(cert)) {
    fs.mkdirSync(CERT_DIR, { recursive: true });
    execSync(
      `openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=visionairy" ` +
        `-keyout "${key}" -out "${cert}"`,
      { stdio: 'ignore' }
    );
  }
  return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
};

const ROUTES = { '/': 'index.html', '/headset': 'headset.html', '/graph': 'graph.html' };

const server = https.createServer(loadCert(), (req, res) => {
  const url = new URL(req.url, 'https://x');
  if (url.pathname === '/ips.json') {
    res.writeHead(200, { 'Content-Type': MIME['.json'] });
    return res.end(JSON.stringify({ ips: lanIPs(), port: PORT }));
  }
  const rel = ROUTES[url.pathname] || url.pathname.slice(1);
  const file = path.normalize(path.join(PUBLIC, rel));
  if (!file.startsWith(PUBLIC)) return res.writeHead(403).end();
  fs.readFile(file, (err, data) => {
    if (err) return res.writeHead(404).end('not found');
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});

// --- Beacon link -----------------------------------------------------------
// The beacon sends heartbeats (VAB1) and throw/calibration events (VAE1), see esp32/beacon/beacon.ino.
// Two ways in: UDP broadcast on port 4210 when it shares the laptop's Wi-Fi, or the USB cable.

const wss = new WebSocketServer({ server, path: '/ws' });
const appServer = http.createServer((req, res) => res.writeHead(404).end());
const appWss = new WebSocketServer({ server: appServer, path: '/ws' });
const allClients = () => [...wss.clients, ...appWss.clients];
const udp = dgram.createSocket({ type: 'udp4', reuseAddr: true });
let beacon = null; // latest heartbeat + how to reach it
let usb = null; // open serial port, if the beacon is plugged in

function broadcast(msg) {
  const s = JSON.stringify(msg);
  for (const c of allClients()) if (c.readyState === 1) c.send(s);
}

let lastEvent = ''; // each event can arrive twice (UDP resend, or Wi-Fi + USB): drop the repeat
let lastStateAt = 0; // state comes over both links too: keep whichever is newer

function handleLine(text, via) {
  const [tag, ...f] = text.trim().split(' ');
  if (tag === 'VAS1') {
    // VAS1 <uptime> <px> <py> <pz> <qw> <qx> <qy> <qz> <flags>
    const n = f.map(Number);
    if (n[0] <= lastStateAt && lastStateAt - n[0] < 5000) return; // duplicate (rebooted beacons restart at 0)
    lastStateAt = n[0];
    broadcast({ type: 'bstate', t: n[0], p: n.slice(1, 4), q: n.slice(4, 8), still: !!(n[8] & 1), air: !!(n[8] & 2), cal: !!(n[8] & 4) });
  } else if (tag === 'VAB1') {
    const [uptime, rssi, led, imu, accel, spin, airborne, cal, yaw, pitch] = f;
    if (!beacon) console.log(`beacon online via ${via.kind === 'usb' ? 'USB' : via.address}`);
    // Prefer USB for commands when both links are up: it doesn't drop packets
    if (beacon?.via.kind === 'usb' && via.kind !== 'usb' && Date.now() - beacon.seen < 1000) via = beacon.via;
    beacon = {
      via, uptime: +uptime, rssi: +rssi, led,
      imu: imu === '1', accel: +accel, spin: +spin, airborne: airborne === '1',
      cal: cal === '1', yaw: +yaw, pitch: +pitch, seen: Date.now(),
    };
    const { via: _, ...pub } = beacon;
    broadcast({ type: 'beacon', ...pub, link: via.kind });
  } else if (tag === 'VAE1' && text !== lastEvent) {
    lastEvent = text;
    console.log(`beacon: ${f.join(' ')}`);
    const [event, ...rest] = f;
    const n = rest.map(Number);
    if (event === 'launch') {
      // VAE1 launch <uptime> <cal> <vx> <vy> <vz> <dx> <dy> <dz>
      broadcast({ type: 'imu', event, cal: n[1] === 1, v: n.slice(2, 5), d: n.slice(5, 8) });
    } else if (event === 'land') {
      // VAE1 land <uptime> <airtime ms> <impact g> <max spin>
      broadcast({ type: 'imu', event, airtime: n[1], impactG: n[2], spin: n[3] });
    } else if (event === 'cal') {
      broadcast({ type: 'imu', event, result: rest[0] });
    } else if (event === 'zero') {
      broadcast({ type: 'imu', event });
    }
  }
}

function sendToBeacon(cmd) {
  if (!beacon) return;
  if (beacon.via.kind === 'usb' && usb?.isOpen) usb.write(cmd + '\n');
  else if (beacon.via.kind === 'udp') udp.send(cmd, beacon.via.port, beacon.via.address);
}

udp.on('message', (buf, rinfo) => handleLine(buf.toString(), { kind: 'udp', address: rinfo.address, port: rinfo.port }));
udp.on('error', (err) => console.log(`beacon Wi-Fi link unavailable: ${err.message}`));
udp.bind(BEACON_PORT);

// USB: optional, keeps retrying so you can plug the beacon in (or flash it) any time
function connectUsb() {
  let SerialPort, ReadlineParser;
  try {
    ({ SerialPort } = require('serialport'));
    ({ ReadlineParser } = require('@serialport/parser-readline'));
  } catch {
    return console.log('serialport not installed: USB beacon link disabled (run npm install)');
  }
  const retry = () => setTimeout(tryOpen, 2000);
  async function tryOpen() {
    const ports = await SerialPort.list().catch(() => []);
    const hit = ports.find((p) => /usbserial|wchusbserial|SLAB_USBtoUART/.test(p.path));
    if (!hit) return retry();
    const path = hit.path.replace('/dev/tty.', '/dev/cu.'); // cu.* doesn't wait for carrier detect
    // hupcl off + DTR/RTS low so opening the port doesn't reset the ESP32 (and lose its calibration)
    const port = new SerialPort({ path, baudRate: 115200, hupcl: false, autoOpen: false });
    port.open((err) => {
      if (err) return retry(); // busy (e.g. flashing): try again shortly
      port.set({ dtr: false, rts: false }, () => {});
      usb = port;
      console.log(`beacon USB link on ${path}`);
      port.pipe(new ReadlineParser({ delimiter: '\n' })).on('data', (line) => handleLine(line, { kind: 'usb' }));
    });
    port.on('close', () => {
      if (usb === port) usb = null;
      console.log('beacon USB link closed');
      retry();
    });
    port.on('error', () => {});
  }
  tryOpen();
}
connectUsb();

setInterval(() => {
  if (beacon && Date.now() - beacon.seen > 3000) {
    console.log('beacon lost');
    beacon = null;
    broadcast({ type: 'beacon', online: false });
  }
}, 1000);

function onClient(ws) {
  if (beacon) {
    const { via, ...pub } = beacon;
    ws.send(JSON.stringify({ type: 'beacon', ...pub, link: via.kind }));
  }
  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === 'cmd' && ['cal', 'zero', 'on', 'off', 'blink'].includes(msg.cmd)) sendToBeacon(msg.cmd);
      // The headset phone's own tracked pose, relayed to the 3D graph on the laptop
      if (msg.type === 'phone') broadcast(msg);
      // The drone's position as the iOS app sees it (camera + LiDAR), for drone/drone.py to follow a path
      if (msg.type === 'drone') broadcast(msg);
      // A throw's arc from the iOS app, saved for the drone to fly later
      if (msg.type === 'path') {
        const dir = path.join(__dirname, 'paths');
        fs.mkdirSync(dir, { recursive: true });
        const file = path.join(dir, `throw-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
        fs.writeFileSync(file, JSON.stringify({ savedAt: new Date().toISOString(), ...msg }, null, 1));
        console.log(`saved ${path.relative(__dirname, file)} (${msg.points.length} points, ${msg.airtime.toFixed(2)} s)`);
        broadcast(msg);
      }
    } catch {}
  });
}
wss.on('connection', onClient);
// Keep-alive: the iOS app treats 2 s of silence as "no server", and with no beacon attached nothing else is sent
setInterval(() => broadcast({ type: 'tick' }), 1000);
appWss.on('connection', onClient);
appServer.listen(APP_PORT, '0.0.0.0');

server.listen(PORT, '0.0.0.0', () => {
  console.log('\n  VisionAiry running. On the headset phone, open (or scan the QR code at https://localhost:' + PORT + '):\n');
  for (const ip of lanIPs()) console.log(`    https://${ip}:${PORT}/headset`);
  console.log(`\n  On this laptop:  https://localhost:${PORT}/headset  (webcam, untick split-screen)`);
  console.log('  (Accept the certificate warning.)\n');
});
