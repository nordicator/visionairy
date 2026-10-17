// VisionAiry relay server.
// Room cameras push JPEG frames over WebSocket; headsets receive the latest frame from every camera.
// HTTPS is required: phones refuse camera + gyro access on plain http.

const https = require('https');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execSync } = require('child_process');
const { WebSocketServer } = require('ws');

const PORT = process.env.PORT || 8443;
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

const ROUTES = { '/': 'index.html', '/camera': 'camera.html', '/headset': 'headset.html' };

const server = https.createServer(loadCert(), (req, res) => {
  const url = new URL(req.url, 'https://x');
  const rel = ROUTES[url.pathname] || url.pathname.slice(1);
  const file = path.normalize(path.join(PUBLIC, rel));
  if (!file.startsWith(PUBLIC)) return res.writeHead(403).end();
  fs.readFile(file, (err, data) => {
    if (err) return res.writeHead(404).end('not found');
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});

// --- WebSocket relay -------------------------------------------------------

const wss = new WebSocketServer({ server, path: '/ws' });
const cameras = new Map(); // slot (0-255) -> { ws, info }
const viewers = new Set();

function cameraList() {
  return [...cameras.entries()].map(([slot, c]) => ({ slot, ...c.info }));
}

function broadcastList() {
  const msg = JSON.stringify({ type: 'cameras', list: cameraList() });
  for (const v of viewers) if (v.readyState === 1) v.send(msg);
}

function freeSlot() {
  for (let i = 0; i < 256; i++) if (!cameras.has(i)) return i;
  return -1;
}

wss.on('connection', (ws, req) => {
  const role = new URL(req.url, 'https://x').searchParams.get('role');

  if (role === 'viewer') {
    viewers.add(ws);
    ws.send(JSON.stringify({ type: 'cameras', list: cameraList() }));
    ws.on('close', () => viewers.delete(ws));
    return;
  }

  if (role === 'camera') {
    const slot = freeSlot();
    if (slot < 0) return ws.close();
    cameras.set(slot, { ws, info: { name: `CAM ${slot + 1}`, heading: 0, pitch: 0, fov: 60 } });
    ws.send(JSON.stringify({ type: 'slot', slot }));
    console.log(`camera ${slot} connected`);

    ws.on('message', (data, isBinary) => {
      if (!isBinary) {
        // Config update from the camera page (name / heading / pitch / fov)
        try {
          const msg = JSON.parse(data.toString());
          if (msg.type === 'info') {
            const c = cameras.get(slot);
            c.info = { ...c.info, ...msg.info };
            broadcastList();
          }
        } catch {}
        return;
      }
      // Frame: prefix with the slot byte and fan out. Drop frames for slow viewers
      // so latency stays low instead of building a backlog.
      const out = Buffer.concat([Buffer.from([slot]), data]);
      for (const v of viewers) {
        if (v.readyState === 1 && v.bufferedAmount < 512 * 1024) v.send(out, { binary: true });
      }
    });

    ws.on('close', () => {
      cameras.delete(slot);
      console.log(`camera ${slot} disconnected`);
      broadcastList();
    });
    broadcastList();
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log('\n  VisionAiry running. On each phone, open:\n');
  for (const ip of lanIPs()) console.log(`    https://${ip}:${PORT}`);
  console.log('\n  (Accept the certificate warning. All devices must be on the same Wi-Fi.)\n');
});
