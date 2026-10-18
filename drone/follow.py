"""Follow a throw's path: fly the drone along the arc the iOS app measured, steered by where the app sees it.

The iOS app (ios/, DRONE button, tap the drone) finds the drone with the camera + LiDAR and sends its
position through server.js, in the same frame as the latest throw path: origin = release spot, Z up,
Y = throw direction, X right, metres. This file reads both from the server and turns the error into sticks.

  1. heading: the drone has no compass and we never yaw it, so first it pushes forward briefly and
     watches which way it went. That gives the stick axes in the path frame.
  2. approach: fly to the start of the path.
  3. follow: a target point slides along the path at SPEED (the real throw is far too fast for a toy
     drone, so it's the same shape, slower) and waits for the drone if it falls behind.
  4. done: drone.py lands it. It also lands if the phone loses the drone for LOST_AFTER seconds.

The path ends where the bag hit the floor, so the drone's height is kept FLOOR_CLEARANCE above that.
"""
import base64
import glob
import json
import math
import os
import socket
import struct
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS_DIR = os.path.join(HERE, "..", "paths")
SERVER_PORT = 8080        # server.js's plain WebSocket port (the one the iOS app uses)

SPEED = 0.25              # m/s along the path
FLOOR_CLEARANCE = 0.30    # m above the landing point
LAG_MAX = 0.25            # m behind the target before the target waits
ARRIVE_TOL = 0.12         # m from the start / end counted as there
ARRIVE_HOLD = 0.5         # s within ARRIVE_TOL
STALE = 0.4               # s without a drone fix: hold still
LOST_AFTER = 3.0          # s without a drone fix: give up (drone.py lands)
APPROACH_TIMEOUT = 20.0   # s to reach the start of the path

PROBE_SETTLE = 1.0        # s hovering before the heading probe
PROBE_PUSH = 0.7          # s of forward stick
PROBE_STICK = 35.0        # % forward stick for the probe
PROBE_COAST = 1.0         # s after the push before measuring
PROBE_MIN_MOVE = 0.08     # m it has to move for the heading to count

# Stick shaping, like target.py: toy drones ignore small inputs
KP = 90.0                 # % stick per metre of error
KD = 60.0                 # % stick per m/s of speed (brakes)
MIN_STICK = 12.0          # % stick once outside the dead zone
MAX_STICK = 45.0          # % stick, cap (X / Y)
MAX_LIFT = 60.0           # % throttle, cap (Z)
DEAD_ZONE = 0.04          # m of error ignored


class ServerFeed:
    """WebSocket client for server.js: the latest drone fix and every throw path it has seen.

    Stdlib only (the laptop is on the drone's Wi-Fi with no internet, so nothing to pip install there)."""

    def __init__(self, host="127.0.0.1", port=SERVER_PORT, log=print):
        self.host, self.port, self.log = host, port, log
        self.lock = threading.Lock()
        self.paths = {}           # path id -> [(t, x, y, z)]
        self.latest_path = None   # id of the newest path
        self.fix = None           # (received at, path id, (x, y, z)) or None
        self.connected = False
        self._load_saved()
        threading.Thread(target=self._run, daemon=True).start()

    def _load_saved(self):
        # Paths saved earlier this session. Only ones with an id can match the phone's drone fixes.
        for f in sorted(glob.glob(os.path.join(PATHS_DIR, "throw-*.json"))):
            try:
                with open(f) as fh:
                    self._add_path(json.load(fh))
            except (OSError, ValueError):
                pass

    def _add_path(self, msg):
        if msg.get("id") and msg.get("points"):
            with self.lock:
                self.paths[msg["id"]] = [tuple(p) for p in msg["points"]]
                self.latest_path = msg["id"]

    def drone(self):
        """(seconds since the last fix, path id, (x, y, z)) or None"""
        with self.lock:
            if self.fix is None:
                return None
            at, pid, p = self.fix
        return time.time() - at, pid, p

    def path(self, pid):
        with self.lock:
            return self.paths.get(pid)

    def _run(self):
        while True:
            try:
                self._session()
            except (OSError, ValueError, ConnectionError):
                pass
            if self.connected:
                self.log("lost the VisionAiry server – retrying")
            self.connected = False
            time.sleep(1)

    def _session(self):
        sock = socket.create_connection((self.host, self.port), timeout=3)
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall((f"GET /ws HTTP/1.1\r\nHost: {self.host}:{self.port}\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        f = sock.makefile("rb")
        if b" 101 " not in f.readline():
            raise ConnectionError("no websocket upgrade")
        while f.readline() not in (b"\r\n", b""):
            pass
        sock.settimeout(5)            # the server ticks every second
        self.connected = True
        self.log(f"VisionAiry server connected ({self.host}:{self.port})")
        while True:
            op, data = self._frame(f)
            if op == 8:
                return
            if op == 9:               # ping -> pong (client frames must be masked)
                mask = os.urandom(4)
                sock.sendall(bytes([0x8A, 0x80 | len(data)]) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
            elif op == 1:
                self._on_message(json.loads(data))

    @staticmethod
    def _frame(f):
        """One whole message (opcode, payload); continuation frames are joined."""
        op, payload = None, b""
        while True:
            head = f.read(2)
            if len(head) < 2:
                raise ConnectionError("closed")
            fin, opcode, n = head[0] & 0x80, head[0] & 0x0F, head[1] & 0x7F
            if n == 126:
                n = struct.unpack(">H", f.read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", f.read(8))[0]
            mask = f.read(4) if head[1] & 0x80 else None
            data = f.read(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if opcode >= 8:
                return opcode, data
            op = op if opcode == 0 else opcode
            payload += data
            if fin:
                return op, payload

    def _on_message(self, msg):
        kind = msg.get("type")
        if kind == "path":
            self._add_path(msg)
            self.log(f"new throw path {msg.get('id')}: {len(msg.get('points', []))} points")
        elif kind == "drone":
            with self.lock:
                if msg.get("ok") and msg.get("p"):
                    self.fix = (time.time(), msg.get("path"), tuple(msg["p"]))


class PathFollower:
    def __init__(self, feed):
        self.feed = feed
        self.state = "off"        # off | heading | approach | follow | done | lost | failed
        self.why = ""
        self.path = None          # resampled [(x, y, z)], evenly spaced
        self.step = 0.01          # m between resampled points
        self.s = 0.0              # target's distance along the path
        self.heading = None       # (fx, fy): the drone's forward stick direction in the path frame
        self.vel = (0.0, 0.0, 0.0)
        self.last_fix = None      # (time, (x, y, z)) for the speed estimate
        self.phase_start = self.there_since = self.lost_since = None
        self.probe_from = None

    @property
    def active(self):
        return self.state in ("heading", "approach", "follow")

    def start(self, now):
        """Begin following the newest path the phone is tracking the drone against. Returns an error or None."""
        fix = self.feed.drone()
        if not self.feed.connected:
            return "not connected to the VisionAiry server (npm start)"
        if fix is None or fix[0] > STALE:
            return "phone isn't tracking the drone (DRONE button, tap it)"
        _, pid, _ = fix
        points = self.feed.path(pid) if pid else None
        if not points:
            return "no throw path yet (THROW START / LANDED in the app)"
        self.path = self._resample(points)
        self.s = 0.0
        self.heading = None
        self.vel = (0.0, 0.0, 0.0)
        self.last_fix = None
        self.why = ""
        self._enter("heading", now)
        return None

    def cancel(self, why="off"):
        self.state = why

    def _enter(self, state, now):
        self.state = state
        self.phase_start = now
        self.there_since = None
        self.probe_from = None

    def _resample(self, points):
        """Evenly spaced points along the arc, with the height floored above the landing spot"""
        floor = points[-1][3] + FLOOR_CLEARANCE
        pts = [(x, y, max(z, floor)) for _, x, y, z in points]
        out = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            d = math.dist(a, b)
            n = max(1, int(d / self.step))
            out += [tuple(a[i] + (b[i] - a[i]) * k / n for i in range(3)) for k in range(1, n + 1)]
        return out

    def status(self):
        if self.state == "heading":
            return "FOLLOW: learning which way is forward"
        if self.state == "approach":
            return "FOLLOW: flying to the start of the path"
        if self.state == "follow":
            return f"FOLLOW: {min(100, round(100 * self.s / max(self.step, self.length)))}% of the path"
        return ""

    @property
    def length(self):
        return (len(self.path) - 1) * self.step if self.path else 0.0

    def _at(self, s):
        return self.path[min(len(self.path) - 1, max(0, round(s / self.step)))]

    def command(self, now):
        """Stick levels (% per axis, drone frame) for this moment"""
        hold = {"x": 0.0, "y": 0.0, "z": 0.0}
        if not self.active:
            return hold
        fix = self.feed.drone()
        if fix is None or fix[0] > STALE:
            self.lost_since = self.lost_since or now
            if now - self.lost_since > LOST_AFTER:
                self.why = "the phone lost the drone"
                self.cancel("lost")
            return hold
        self.lost_since = None
        _, _, p = fix
        self._update_speed(now, p)

        if self.state == "heading":
            return self._probe(now, p)

        if self.state == "approach":
            goal = self.path[0]
            if self._arrived(now, p, goal):
                self._enter("follow", now)
            elif now - self.phase_start > APPROACH_TIMEOUT:
                self.why = "couldn't reach the start of the path"
                self.cancel("failed")
                return hold
        else:
            # Slide the target along; it waits whenever the drone falls behind
            if math.dist(p, self._at(self.s)) < LAG_MAX:
                self.s = min(self.length, self.s + SPEED * (now - self.phase_start))
            self.phase_start = now
            goal = self._at(self.s)
            if self.s >= self.length and self._arrived(now, p, goal):
                self.cancel("done")
                return hold
        return self._steer(p, goal)

    def _probe(self, now, p):
        t = now - self.phase_start
        if t < PROBE_SETTLE:
            self.probe_from = p
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        if t < PROBE_SETTLE + PROBE_PUSH:
            return {"x": 0.0, "y": PROBE_STICK, "z": 0.0}
        if t < PROBE_SETTLE + PROBE_PUSH + PROBE_COAST:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        dx, dy = p[0] - self.probe_from[0], p[1] - self.probe_from[1]
        moved = math.hypot(dx, dy)
        if moved < PROBE_MIN_MOVE:
            self.why = f"it only moved {moved * 100:.0f} cm on the test push – can't tell which way it faces"
            self.cancel("failed")
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        self.heading = (dx / moved, dy / moved)
        self._enter("approach", now)
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    def _update_speed(self, now, p):
        if self.last_fix and now - self.last_fix[0] > 0.02:
            dt = now - self.last_fix[0]
            raw = [(p[i] - self.last_fix[1][i]) / dt for i in range(3)]
            self.vel = tuple(0.7 * v + 0.3 * r for v, r in zip(self.vel, raw))
        if not self.last_fix or now - self.last_fix[0] > 0.02:
            self.last_fix = (now, p)

    def _arrived(self, now, p, goal):
        if math.dist(p, goal) < ARRIVE_TOL:
            self.there_since = self.there_since or now
            return now - self.there_since >= ARRIVE_HOLD
        self.there_since = None
        return False

    def _steer(self, p, goal):
        e = [goal[i] - p[i] for i in range(3)]
        fx, fy = self.heading
        rx, ry = fy, -fx                       # right of forward, seen from above (Z up)
        push = [KP * e[i] - KD * self.vel[i] for i in range(3)]
        fwd = push[0] * fx + push[1] * fy
        right = push[0] * rx + push[1] * ry
        flat = math.hypot(e[0], e[1]) > DEAD_ZONE
        return {"x": shape(right, MAX_STICK) if flat else 0.0,
                "y": shape(fwd, MAX_STICK) if flat else 0.0,
                "z": shape(push[2], MAX_LIFT) if abs(e[2]) > DEAD_ZONE else 0.0}


def shape(v, cap):
    if abs(v) < 1e-6:
        return 0.0
    return math.copysign(min(cap, MIN_STICK + abs(v)), v)
