// XRAY bridge: reads the ESP32 over USB serial and relays every line to the visualizer over WebSocket.
// Browser messages are written back to the ESP32 as commands (c = recalibrate, t<n> = threshold).

const http = require('http');
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');

const HTTP_PORT = process.env.PORT || 8090;
const BAUD = 921600;
const PUBLIC = path.join(__dirname, 'public');

let port = null;
let portPath = process.argv[2] || process.env.SERIAL_PORT || null;
const recentStatus = []; // replayed to new browser tabs

const server = http.createServer((req, res) => {
  const rel = req.url === '/' ? 'index.html' : req.url.split('?')[0].slice(1);
  const file = path.normalize(path.join(PUBLIC, rel));
  if (!file.startsWith(PUBLIC)) return res.writeHead(403).end();
  fs.readFile(file, (err, data) => {
    if (err) return res.writeHead(404).end('not found');
    const type = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }[path.extname(file)];
    res.writeHead(200, { 'Content-Type': `${type || 'application/octet-stream'}; charset=utf-8` });
    res.end(data);
  });
});

const wss = new WebSocketServer({ server });

function broadcast(line) {
  for (const c of wss.clients) if (c.readyState === 1) c.send(line);
}

function status(text) {
  const line = `S ${text}`;
  recentStatus.push(line);
  if (recentStatus.length > 30) recentStatus.shift();
  broadcast(line);
  console.log(line);
}

wss.on('connection', (ws) => {
  for (const s of recentStatus) ws.send(s);
  ws.send(`P ${port && port.isOpen ? portPath : '-'}`);
  ws.on('message', (msg) => {
    const cmd = msg.toString().trim();
    if (/^(c|r|t[0-9.]+)$/.test(cmd) && port && port.isOpen) port.write(cmd + '\n');
  });
});

let lastNote = '';
function noteOnce(text) {
  if (text !== lastNote) status(text);
  lastNote = text;
}

async function findPort() {
  if (portPath) return portPath;
  const ports = await SerialPort.list();
  const hit = ports.find((p) => /usbserial|wchusbserial|SLAB_USBtoUART|usbmodem/.test(p.path));
  // On macOS the cu.* device doesn't wait for carrier detect, tty.* can block.
  return hit && hit.path.replace('/dev/tty.', '/dev/cu.');
}

async function connect() {
  const p = await findPort().catch(() => null);
  if (!p) {
    noteOnce('laptop: no ESP32 serial port found, plug it in (retrying)');
    broadcast('P -');
    return setTimeout(connect, 2000);
  }
  port = new SerialPort({ path: p, baudRate: BAUD }, (err) => {
    if (err) {
      noteOnce(`laptop: can't open ${p}: ${err.message} (retrying)`);
      port = null;
      return setTimeout(connect, 2000);
    }
    lastNote = '';
    status(`laptop: serial open on ${p}`);
    broadcast(`P ${p}`);
  });
  portPath = process.argv[2] || process.env.SERIAL_PORT || p;
  const parser = port.pipe(new ReadlineParser({ delimiter: '\n' }));
  parser.on('data', (raw) => {
    const line = raw.trim();
    if (!line) return;
    if (line.startsWith('S ')) {
      recentStatus.push(line);
      if (recentStatus.length > 30) recentStatus.shift();
      console.log(line);
    }
    broadcast(line);
  });
  port.on('close', () => {
    status('laptop: serial closed, reconnecting');
    broadcast('P -');
    port = null;
    setTimeout(connect, 2000);
  });
  port.on('error', () => {});
}

server.listen(HTTP_PORT, () => {
  console.log(`XRAY visualizer: http://localhost:${HTTP_PORT}`);
  connect();
});
