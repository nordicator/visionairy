#!/usr/bin/env python3
"""Fly an AI pilot command on a simulated drone and draw the path it would take.

  python sim_flight.py "take off, fly a 1 m square, turn 90 degrees right, land"
  python sim_flight.py --wind 0.15 "..."     # add a steady breeze + gusts (m/s) to stress the plan
  python sim_flight.py --model spur-glm-5-3 "..."

The real LLM (from drone/.env) plans the flight through the same tools and Bridge as drone.py --ai;
the stick bytes it would send drive a simple model drone instead of a real one. Saves a picture
(top view + height over time) to media/ and prints where the drone was after each tool call.

The model drone uses the same rough speeds the LLM is told (30% ≈ 0.5 m/s, 50% ≈ 1 m/s,
turn 50% ≈ 90°/s, up 50% ≈ 0.5 m/s) plus momentum, so this checks that the plan hangs together:
does the square close, does it turn the right way, does it climb before it moves. How your real
drone flies is only as close as those numbers; measure it and put the results in ai_notes.md.
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

import ai_pilot
import flightlog
import supervisor
from protocol import CENTER, FLAG_GYRO_CAL, FLAG_LAND, FLAG_STOP, FLAG_TAKEOFF

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(HERE, "media")

SPEED_PER_PCT = 0.025     # m/s per % stick past the dead zone (30% -> 0.5 m/s, 50% -> 1.0 m/s)
STICK_DEAD = 10.0         # % below which a toy drone barely reacts
CLIMB_PER_PCT = 0.01      # m/s per % throttle (50% -> 0.5 m/s)
YAW_MAX = 178.0           # deg/s at full yaw byte, so the AI's 50% turns ~90 deg/s
YAW_DEAD = 24             # yaw bytes 104..152 do nothing
TAU = 0.35                # s, how fast the drone picks up / loses speed (momentum)
HOVER_HEIGHT = 1.0        # m, where take off climbs to
TAKEOFF_RATE = 0.7        # m/s
LAND_RATE = 0.5           # m/s

# what the simulated camera sees (stands in for drift.py and obstacles.py)
FW_PER_MPS = 0.4          # frame-widths/s of measured slide per m/s of real sideways motion
FW_PER_MPS_FWD = 0.2      # forward motion is measured from zoom, which is weaker
FW_PER_RAD = 0.8          # a turn makes the scene slide too: frame-widths per radian (~70 deg view)
EST_TAU = 0.6             # s, smoothing, like drift.SMOOTH_TAU
FOV_H = 70.0              # deg, horizontal field of view
OBST_SIZE = 0.5           # m, obstacles are boxes this big
COLLIDE = 0.3             # m, closer than this to an obstacle's centre = crash


def pct(byte):
    return (byte - CENTER) / 127 * 100


class SimDrone:
    """Kinematic toy quadcopter. Heading 0 = the way the nose pointed at take off, + = clockwise."""

    def __init__(self, wind=0.0, seed=1, wind_dir=None, spikes=(), yaw_drift=0.0, invert=(), alt_drift=0.0):
        self.x = self.y = self.z = 0.0      # m: x = right, y = forward (of the take-off heading), z = up
        self.heading = 0.0
        self.vx = self.vy = 0.0             # world-frame m/s
        self.mode = "ground"                # ground | takeoff | flying | landing | falling
        rng = random.Random(seed)
        a = rng.uniform(0, 2 * math.pi) if wind_dir is None else math.radians(90 - wind_dir)
        self.wind = (wind * math.cos(a), wind * math.sin(a))
        self.gust_amp, self.rng, self.gust = wind * 0.5, rng, [0.0, 0.0]
        self.track = []                     # (t, x, y, z, heading, mode)
        self.t = None
        self.yaw_rate = 0.0                 # deg/s
        self.spikes = list(spikes)          # (s after take off, vx, vy, duration): a sudden push, m/s
        self.yaw_drift = yaw_drift          # deg/s the drone turns by itself (uncalibrated gyro)
        self.invert = set(invert)           # stick axes ("x", "y", "yaw") this drone's firmware reverses
        self.alt_drift = alt_drift          # m/s it climbs (+) or sinks (-) at centred throttle (no altitude hold)
        self.vz = 0.0
        self.takeoff_t = None

    def step(self, s, now):
        dt = 0.0 if self.t is None else min(0.2, now - self.t)
        self.t = now
        if s.flags & FLAG_STOP and self.mode != "ground":
            self.mode = "falling"
        elif s.flags & FLAG_LAND and self.mode in ("flying", "takeoff"):
            self.mode = "landing"
        elif s.flags & FLAG_TAKEOFF and self.mode == "ground":
            self.mode = "takeoff"
            self.takeoff_t = now
        elif s.flags & FLAG_GYRO_CAL and self.mode == "ground" and self.yaw_drift:
            self.yaw_drift = 0.0            # calibrating on the ground removes the gyro bias
            flightlog.event("sim_gyro_fixed")
        air = self.mode in ("takeoff", "flying", "landing")

        # stick -> wanted body velocity, then into the world frame by heading
        def speed(p):
            return math.copysign(max(0.0, abs(p) - STICK_DEAD) * SPEED_PER_PCT, p)
        sx = -1 if "x" in self.invert else 1
        sy = -1 if "y" in self.invert else 1
        fwd, right = speed(sy * pct(s.pitch)), speed(sx * pct(s.roll))
        h = math.radians(self.heading)
        want_x = right * math.cos(h) + fwd * math.sin(h)
        want_y = -right * math.sin(h) + fwd * math.cos(h)
        if not air:
            want_x = want_y = 0.0
        else:
            for i in (0, 1):    # gusts: a slow random walk around the steady wind
                self.gust[i] += self.rng.gauss(0, self.gust_amp) * dt - self.gust[i] * dt
            want_x += self.wind[0] + self.gust[0]
            want_y += self.wind[1] + self.gust[1]
            since = now - (self.takeoff_t or now)
            for t0, px, py, dur in self.spikes:
                if t0 <= since < t0 + dur:
                    want_x += px
                    want_y += py
        k = 1 - math.exp(-dt / TAU) if dt else 0.0
        self.vx += (want_x - self.vx) * k
        self.vy += (want_y - self.vy) * k
        self.x += self.vx * dt
        self.y += self.vy * dt

        o = (s.yaw - CENTER) * (-1 if "yaw" in self.invert else 1)
        self.yaw_rate = self.yaw_drift if air else 0.0
        if air and abs(o) > YAW_DEAD:
            self.yaw_rate += math.copysign((abs(o) - YAW_DEAD) / (127 - YAW_DEAD) * YAW_MAX, o)
        self.heading += self.yaw_rate * dt

        z0 = self.z
        if self.mode == "takeoff":
            self.z = min(HOVER_HEIGHT, self.z + TAKEOFF_RATE * dt)
            if self.z >= HOVER_HEIGHT:
                self.mode = "flying"
        elif self.mode == "flying":
            self.z = max(0.05, self.z + (pct(s.throttle) * CLIMB_PER_PCT + self.alt_drift) * dt)
        elif self.mode in ("landing", "falling"):
            self.z = max(0.0, self.z - (LAND_RATE if self.mode == "landing" else 3.0) * dt)
            if self.z <= 0:
                self.mode = "ground"
                self.vx = self.vy = 0.0
        self.vz = (self.z - z0) / dt if dt else 0.0
        self.track.append((now, self.x, self.y, self.z, self.heading, self.mode))

    def pose(self):
        return (f"x {self.x:+.2f} m  y {self.y:+.2f} m  z {self.z:.2f} m  "
                f"heading {self.heading:+.0f}°  {self.mode}")


    def body_velocity(self):
        """(right, forward) m/s relative to where the nose points."""
        h = math.radians(self.heading)
        return (self.vx * math.cos(h) - self.vy * math.sin(h), self.vx * math.sin(h) + self.vy * math.cos(h))


class SimCamera:
    """perceive(now) for DryRun: the drift estimate and tracked objects the real video would give.

    The drift estimate is the drone's real body velocity scaled into frame-widths/s, smoothed, plus
    the false sideways slide a turn causes. Obstacles are boxes at fixed spots in the room, seen
    when inside the field of view, sized by distance, with time-to-contact from the closing speed."""

    def __init__(self, sim, lock, obstacles, sign_bug=False, video_drop=None, wall_y=None, noise=0.004, seed=1,
                 latency=0.0, depth_scale=1.0, depth_noise=0.05, depth=True):
        self.sim, self.lock, self.obstacles = sim, lock, obstacles
        self.sign_bug = sign_bug            # the slide estimate's x comes out reversed
        self.video_drop = video_drop        # (s after take off, duration): no video at all
        self.wall_y = wall_y                # a blank wall across the room at this y: no texture near it
        self.noise, self.rng = noise, random.Random(seed + 7)
        self.vel = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.latency = latency              # s the video lags behind reality (toy drones: 0.2-0.5)
        self.depth = depth                  # camera depth (depth.py) available
        self.depth_scale = depth_scale      # its error: 1.5 = says everything is 50% farther than it is
        self.depth_noise = depth_noise      # ... plus this much random relative error per reading
        self.ahead_now = None               # (m straight ahead, close) as obstacles.ahead_info() gives
        self.history = deque()              # (time, raw estimate, objects) waiting out the latency
        self.t = None
        self.collisions = []                # (time, obstacle index, x, y)
        self.last_pose = 0.0

    def _dropped(self, now):
        t0 = self.sim.takeoff_t
        if not self.video_drop or t0 is None:
            return None
        start = t0 + self.video_drop[0]
        return now - start if start <= now < start + self.video_drop[1] else None

    def age(self, now):
        d = self._dropped(now)
        return 0.05 if d is None else d

    def __call__(self, now):
        with self.lock:
            sim = self.sim
            right, fwd = sim.body_velocity()
            air = sim.mode in ("takeoff", "flying", "landing") and sim.z > 0.2
            raw = {"x": right * FW_PER_MPS + math.radians(sim.yaw_rate) * FW_PER_RAD,
                   "y": fwd * FW_PER_MPS_FWD, "z": sim.vz * FW_PER_MPS}
            if self.sign_bug:
                raw["x"] = -raw["x"]
            for a in ("x", "y", "z"):
                raw[a] += self.rng.gauss(0, self.noise)
            blank = False
            if self.wall_y is not None:
                if sim.y > self.wall_y - 0.15 and not any(c[1] == "wall" for c in self.collisions):
                    self.collisions.append((now, "wall", sim.x, sim.y))
                    flightlog.event("sim_collision", obstacle="wall", x=round(sim.x, 2), y=round(sim.y, 2))
                    print(f"  !!! COLLISION with the wall at y {sim.y:+.2f}", flush=True)
                facing = abs(((sim.heading + 180) % 360) - 180) < 40
                blank = facing and self.wall_y - sim.y < 0.8
            objects = self._objects(sim, fwd, now) if air else []
            ahead = self._ahead(sim, objects) if air and self.depth else None
            self.history.append((now, raw, objects, ahead))     # what the camera sees arrives late
            while len(self.history) > 1 and self.history[1][0] <= now - self.latency:
                self.history.popleft()
            _, raw, objects, self.ahead_now = self.history[0]
            dt = 0.0 if self.t is None else now - self.t
            self.t = now
            k = 1 - math.exp(-dt / EST_TAU) if dt > 0 else 0.0
            for a in self.vel:
                self.vel[a] += k * (raw[a] - self.vel[a])
            valid = air
            if self._dropped(now) is not None:
                valid, objects, self.ahead_now = False, [], None
            elif blank:
                valid = False
            if now - self.last_pose >= 0.2:
                self.last_pose = now
                flightlog.event("sim_pose", x=round(sim.x, 3), y=round(sim.y, 3), z=round(sim.z, 3),
                                heading=round(sim.heading, 1), mode=sim.mode,
                                v_right=round(right, 3), v_fwd=round(fwd, 3))
            return dict(self.vel), valid, objects

    def _objects(self, sim, v_fwd, now):
        h = math.radians(sim.heading)
        out = []
        for i, (ox, oy) in enumerate(self.obstacles):
            dx, dy = ox - sim.x, oy - sim.y
            dist = math.hypot(dx, dy)
            if dist < COLLIDE and sim.z > 0.2 and not any(c[1] == i for c in self.collisions):
                self.collisions.append((now, i, sim.x, sim.y))
                flightlog.event("sim_collision", obstacle=i + 1, x=round(sim.x, 2), y=round(sim.y, 2))
                print(f"  !!! COLLISION with obstacle #{i + 1} at x {sim.x:+.2f} y {sim.y:+.2f}", flush=True)
            f = dx * math.sin(h) + dy * math.cos(h)
            r = dx * math.cos(h) - dy * math.sin(h)
            if f < 0.15:
                continue
            bearing = math.degrees(math.atan2(r, f))
            if abs(bearing) > FOV_H / 2:
                continue
            u = 0.5 + bearing / FOV_H
            bh = min(0.95, OBST_SIZE / (0.83 * f))
            bw = min(0.95, OBST_SIZE / (1.4 * f))
            ttc = f / v_fwd if v_fwd > 0.05 and f / v_fwd < 6 else None
            box = (max(0.0, u - bw / 2), 0.5 - bh / 2, min(1.0, u + bw / 2), 0.5 + bh / 2)
            in_path = box[2] > 0.3 and box[0] < 0.7
            o = {"id": i + 1, "label": "box", "box": box, "ttc": ttc, "lost": False, "dist": None,
                 "threat": in_path and (bh >= 0.45 or (ttc is not None and ttc < 2.0))}
            if self.depth:                  # what depth.py would say: the near surface, with its error
                d = self._depth_reading(f - OBST_SIZE / 2)
                o.update(dist=d, dist_safe=d * 0.8, dist_fresh=True, bearing=bearing,
                         xyz=(r * d / max(f, 0.01), 0.0, d))
                o["threat"] = in_path and (d * 0.8 < 0.8 or (ttc is not None and ttc < 2.0))   # obstacles.is_threat
            out.append(o)
        return out

    def _depth_reading(self, true_m):
        return max(0.05, true_m * self.depth_scale * (1 + self.rng.gauss(0, self.depth_noise)))

    def _ahead(self, sim, objects):
        """obstacles.ahead_info(): nearest thing straight ahead per depth, and whether it's close."""
        near = [o["dist"] for o in objects if o.get("dist") is not None and abs(o["bearing"]) < 12]
        if self.wall_y is not None and abs(((sim.heading + 180) % 360) - 180) < 40:
            near.append(self._depth_reading(self.wall_y - sim.y))
        if not near:
            near.append(self._depth_reading(4.0))           # the far side of the room
        d = min(near)
        return d, d * 0.8 < 0.8

    def ahead(self, now):
        return self.ahead_now


# ---------------- drawing ----------------
BG, GRID, AXIS, TEXT = (250, 250, 248), (226, 226, 222), (180, 180, 175), (40, 40, 40)
START, END = (60, 160, 60), (40, 40, 200)


def heat(f):
    """0..1 -> blue..orange, for time along the path."""
    c = cv2.applyColorMap(np.uint8([[round(f * 255)]]), cv2.COLORMAP_PLASMA)[0, 0]
    return tuple(int(v) for v in c)


def render(sim, events, title, path, obstacles=(), collisions=()):
    first = next(i for i, p in enumerate(sim.track) if p[5] != "ground")
    tr = sim.track[max(0, first - 10):]                 # skip the wait for the LLM before take off
    events = [(t, label) for t, label in events if t >= tr[0][0]]
    W, H = 1300, 760
    img = np.full((H, W, 3), BG, np.uint8)
    put = lambda s, p, sc=0.5, c=TEXT, th=1: cv2.putText(img, s, p, cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)
    put(title[:110], (24, 34), 0.62, TEXT, 1)

    # top view
    ox, oy, size = 24, 60, 640
    xs = [p[1] for p in tr] + [o[0] for o in obstacles]
    ys = [p[2] for p in tr] + [o[1] for o in obstacles]
    span = max(1.0, max(xs) - min(xs), max(ys) - min(ys)) * 1.25
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    scale = size / span

    def to_px(x, y):
        return int(ox + size / 2 + (x - cx) * scale), int(oy + size / 2 - (y - cy) * scale)
    cv2.rectangle(img, (ox, oy), (ox + size, oy + size), AXIS, 1)
    step = 0.5 if span < 6 else 1.0
    g = math.floor((cx - span / 2) / step) * step
    while g < cx + span / 2:
        x0 = to_px(g, 0)[0]
        if ox < x0 < ox + size:
            cv2.line(img, (x0, oy), (x0, oy + size), GRID, 1)
        g += step
    g = math.floor((cy - span / 2) / step) * step
    while g < cy + span / 2:
        y0 = to_px(0, g)[1]
        if oy < y0 < oy + size:
            cv2.line(img, (ox, y0), (ox + size, y0), GRID, 1)
        g += step
    put(f"top view  (grid {step:g} m, up = the way the nose faced at take off)", (ox, oy + size + 22), 0.45)

    for i, (ox, oy) in enumerate(obstacles):
        a, b = to_px(ox - OBST_SIZE / 2, oy + OBST_SIZE / 2), to_px(ox + OBST_SIZE / 2, oy - OBST_SIZE / 2)
        cv2.rectangle(img, a, b, (150, 150, 235), -1)
        cv2.rectangle(img, a, b, (60, 60, 200), 1)
        put(f"obstacle #{i + 1}", (a[0], a[1] - 5), 0.4, (60, 60, 200))
    t0, t1 = tr[0][0], tr[-1][0]
    for a, b in zip(tr, tr[1:]):
        if a[5] == "ground" and b[5] == "ground":
            continue
        cv2.line(img, to_px(a[1], a[2]), to_px(b[1], b[2]), heat((a[0] - t0) / max(1e-6, t1 - t0)), 3, cv2.LINE_AA)
    last = -1e9
    for t, x, y, z, hd, mode in tr:     # nose arrows every second while flying
        if mode != "ground" and t - last >= 1.0:
            last = t
            p = to_px(x, y)
            h = math.radians(hd)
            q = (int(p[0] + 18 * math.sin(h)), int(p[1] - 18 * math.cos(h)))
            cv2.arrowedLine(img, p, q, (90, 90, 90), 1, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(img, to_px(0, 0), 8, START, 2, cv2.LINE_AA)
    put("start", (to_px(0, 0)[0] + 11, to_px(0, 0)[1] + 4), 0.45, START)
    e = to_px(tr[-1][1], tr[-1][2])
    cv2.drawMarker(img, e, END, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
    put("end", (e[0] + 11, e[1] - 6), 0.45, END)
    for _, i, cx_, cy_ in collisions:
        q = to_px(cx_, cy_)
        cv2.drawMarker(img, q, (0, 0, 255), cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
        put("CRASH", (q[0] + 12, q[1] + 4), 0.5, (0, 0, 255), 2)
    for i, (t, label) in enumerate(events):            # number each tool call on the path
        p = min(tr, key=lambda r: abs(r[0] - t))
        q = to_px(p[1], p[2])
        cv2.circle(img, q, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, q, 9, TEXT, 1, cv2.LINE_AA)
        put(str(i + 1), (q[0] - 4 - 4 * (i >= 9), q[1] + 4), 0.38)

    # height over time
    hx, hy, hw, hh = 700, 60, 576, 260
    cv2.rectangle(img, (hx, hy), (hx + hw, hy + hh), AXIS, 1)
    zmax = max(1.5, max(p[3] for p in tr) * 1.2)
    for zz in np.arange(0.5, zmax, 0.5):
        y0 = int(hy + hh - zz / zmax * hh)
        cv2.line(img, (hx, y0), (hx + hw, y0), GRID, 1)
        put(f"{zz:g} m", (hx + 4, y0 - 3), 0.36, AXIS)
    pts = [(int(hx + (t - t0) / max(1e-6, t1 - t0) * hw), int(hy + hh - z / zmax * hh)) for t, _, _, z, _, _ in tr]
    cv2.polylines(img, [np.array(pts, np.int32)], False, (180, 90, 40), 2, cv2.LINE_AA)
    put(f"height over time (0 - {t1 - t0:.1f} s)", (hx, hy + hh + 22), 0.45)

    # summary
    yy = hy + hh + 60
    d = math.hypot(tr[-1][1], tr[-1][2])
    for line in (f"ended {d:.2f} m from the start, heading {tr[-1][4]:+.0f} deg, {tr[-1][5]}",
                 f"furthest from start {max(math.hypot(p[1], p[2]) for p in tr):.2f} m, "
                 f"highest {max(p[3] for p in tr):.2f} m",
                 f"wind {math.hypot(*sim.wind):.2f} m/s" if any(sim.wind) else "no wind",
                 f"COLLISIONS: {len(collisions)}" if collisions else ("no collisions" if obstacles else "")):
        put(line, (hx, yy), 0.5)
        yy += 24
    yy += 10
    for i, (t, label) in enumerate(events[:16]):
        put(f"{i + 1:>2}  {t - t0:5.1f}s  {label[:70]}", (hx, yy), 0.42)
        yy += 20
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


class ScriptLLM:
    """Stands in for the LLM: replays fixed tool calls, one turn each, then says done."""
    model = base = "scripted"
    chain, last_model = ["scripted"], "scripted"
    log = staticmethod(print)

    def __init__(self, calls):
        self.calls = list(calls)

    def chat(self, messages, tools):
        if not self.calls:
            return {"content": "done (scripted)"}
        name, args = self.calls.pop(0)
        return {"content": None, "tool_calls": [{"id": "s", "type": "function",
                                                 "function": {"name": name, "arguments": json.dumps(args)}}]}


def _key_events(path):
    """The flight log's safety events, for the JSON summary."""
    keep = {"guard_stop", "auto_land", "hold_axis_off", "hold_axis_flip", "hold_tune", "hold_escalate", "alert",
            "plan_end", "tool"}
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["ev"] in keep:
                r = {k: v for k, v in r.items() if k in ("rel", "ev", "why", "axis", "cap", "gain", "text",
                                                       "result", "name", "key")}
                if r["ev"] == "tool":
                    r = {"rel": r["rel"], "ev": "tool", "name": r.get("name")}
                out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="+", help="what to tell the AI pilot")
    ap.add_argument("--model", help="override SPAR_MODEL")
    ap.add_argument("--wind", type=float, default=0.0, help="steady breeze + gusts in m/s (default 0)")
    ap.add_argument("--seed", type=int, default=1, help="wind direction / gust seed")
    ap.add_argument("--max-stick", type=float, default=60.0, help="stick cap in %% (default 60)")
    ap.add_argument("-o", "--out", help="picture path (default media/sim_<time>.png)")
    ap.add_argument("--obstacle", action="append", default=[], metavar="X,Y",
                    help="a 0.5 m box at X m right, Y m forward of the take-off spot (repeatable)")
    ap.add_argument("--no-guard", action="store_true", help="obstacle guard off")
    ap.add_argument("--no-hold", action="store_true", help="position hold off")
    ap.add_argument("--no-autotrim", action="store_true", help="auto-trim off (it's on in drone.py by default)")
    f = ap.add_argument_group("faults (times are seconds after take off)")
    f.add_argument("--wind-dir", type=float, help="wind blows toward this heading, deg (0 = forward, 90 = right)")
    f.add_argument("--spike", action="append", default=[], metavar="T,VX,VY,DUR",
                   help="sudden push: VX right / VY forward m/s for DUR s from T s (repeatable)")
    f.add_argument("--yaw-drift", type=float, default=0.0, help="the drone turns by itself, deg/s")
    f.add_argument("--video-drop", metavar="T,DUR", help="no video from T s for DUR s")
    f.add_argument("--sign-bug", action="store_true", help="the camera's sideways slide estimate is reversed")
    f.add_argument("--blank-wall", type=float, metavar="Y", help="a plain wall across the room Y m ahead")
    f.add_argument("--noise", type=float, default=0.004, help="slide-estimate noise, frame-widths/s")
    f.add_argument("--latency", type=float, default=0.0, help="video lag, s (toy drones: 0.2-0.5)")
    f.add_argument("--invert", default="", help="stick axes the drone reverses, e.g. x,yaw")
    f.add_argument("--alt-drift", type=float, default=0.0, help="climb (+) / sink (-) at centred throttle, m/s")
    f.add_argument("--depth-scale", type=float, default=1.0,
                   help="camera depth error: 1.5 = says things are 50%% farther than they are")
    f.add_argument("--depth-noise", type=float, default=0.05, help="camera depth random error per reading")
    f.add_argument("--no-depth", action="store_true", help="no camera depth at all")
    f.add_argument("--depth-uncalibrated", action="store_true", help="depth with a guessed lens, never calibrated")
    t = ap.add_argument_group("supervisor / planner settings")
    t.add_argument("--hold-gain", type=float, default=supervisor.HOLD_GAIN)
    t.add_argument("--hold-min", type=float, default=supervisor.HOLD_MIN)
    t.add_argument("--max-flight", type=float, default=supervisor.FLIGHT_MAX_S)
    t.add_argument("--lost-land", type=float, default=supervisor.LOST_LAND_S)
    t.add_argument("--max-distance", type=float, default=3.0)
    t.add_argument("--ceiling", type=float, default=2.0)
    t.add_argument("--no-auto-gyro", action="store_true", help="skip the automatic gyro calibration before take off")
    ap.add_argument("--calibration", metavar="PATH", help="apply a calibration file (from preflight.py), as drone.py does")
    ap.add_argument("--plan", help='scripted tool calls instead of the LLM: JSON [["sequence", {...}], ...], '
                                   'one per turn (the command text is then just a label)')
    ap.add_argument("--json", action="store_true", help="print a JSON summary line at the end")
    args = ap.parse_args()
    command = " ".join(args.command)
    obstacles = [tuple(float(v) for v in o.split(",")) for o in args.obstacle]
    print("flight log:", os.path.relpath(flightlog.LOG.open("sim")), flush=True)
    flightlog.event("session", argv=sys.argv, args=vars(args), sim=True)

    llm = ScriptLLM(json.loads(args.plan)) if args.plan else ai_pilot.LLM(model=args.model)
    bridge = ai_pilot.Bridge(args.max_stick)
    spikes = [tuple(float(v) for v in sp.split(",")) for sp in args.spike]
    sim = SimDrone(args.wind, args.seed, args.wind_dir, spikes, args.yaw_drift,
                   [a for a in args.invert.split(",") if a], args.alt_drift)
    lock = threading.Lock()

    camera = SimCamera(sim, lock, obstacles, args.sign_bug,
                       tuple(float(v) for v in args.video_drop.split(",")) if args.video_drop else None,
                       args.blank_wall, args.noise, args.seed, args.latency, args.depth_scale, args.depth_noise,
                       not args.no_depth)
    sup = supervisor.Supervisor(bridge, guard=not args.no_guard, hold=not args.no_hold, hold_gain=args.hold_gain,
                                hold_min=args.hold_min, lost_land_s=args.lost_land, flight_max_s=args.max_flight)
    perceive = camera
    cal = None
    if args.calibration:
        import calibration
        cal = calibration.load(args.calibration)

        class CalibratedCamera:                     # the estimate as drone.py sees it after calibration
            def __call__(self, now):
                vel, valid, objs = camera(now)
                return calibration.apply_est(vel, cal), valid, objs

            def age(self, now):
                return camera.age(now)

            def ahead(self, now):
                return camera.ahead(now)
        perceive = CalibratedCamera()
    depth_info = {"available": not args.no_depth, "lens_guessed": args.depth_uncalibrated,
                  "cal_samples": 0 if args.depth_uncalibrated else 3, "scale": 1.0, "cal_age_days": 1.0,
                  "status": "depth is off"}
    loop = ai_pilot.DryRun(bridge, quiet=True, sup=sup, perceive=perceive, autotrim=not args.no_autotrim,
                           depth=supervisor.DepthTrust(lambda: depth_info))
    if cal:
        loop.pilot.invert = dict(cal["invert"])
    threading.Thread(target=loop.loop, daemon=True).start()

    def physics():          # the drone side: reads the sticks the radio would send, at 200 Hz
        while True:
            s = loop.pilot.sticks()
            with lock:
                sim.step(s, time.time())
            time.sleep(0.005)
    threading.Thread(target=physics, daemon=True).start()

    events = []

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        if line.startswith("  ") and "(" in line:          # a tool call finished
            with lock:
                events.append((time.time(), line.strip().split(" -> ")[0]))
                print(f"      sim: {sim.pose()}", flush=True)

    chat = llm.chat

    def chat_and_show(messages, tools):            # print the plan in full, the agent log shortens it
        msg = chat(messages, tools)
        for c in msg.get("tool_calls") or []:
            if c["function"]["name"] == "sequence":
                print("  plan:", flush=True)
                for i, st in enumerate(json.loads(c["function"]["arguments"]).get("steps", [])):
                    print(f"    {i + 1:>2}. " + ", ".join(f"{k} {v}" for k, v in st.items()), flush=True)
        return msg
    llm.chat = chat_and_show
    agent = ai_pilot.Agent(llm, bridge, log=log, envelope=ai_pilot.Envelope(args.max_distance, args.ceiling),
                           auto_gyro=not args.no_auto_gyro)
    hud_last = [None]

    def status_lines():         # print what the HUD's AI panel would show, when it changes
        while True:
            h = sup.hud(time.time(), agent)
            line = (f"guard {h['guard'][0]}{': ' + h['guard'][1] if h['guard'][1] else ''} | "
                    f"hold {h['hold'][0]}")
            if line != hud_last[0]:
                hud_last[0] = line
                print(f"      hud: {line}", flush=True)
            time.sleep(0.1)
    threading.Thread(target=status_lines, daemon=True).start()
    print(f"simulating with {llm.model}: {command}", flush=True)
    agent.submit(command)
    time.sleep(0.2)
    while agent.busy or agent.inbox:
        time.sleep(0.1)
    end = time.time() + 6                                # let a landing finish
    while sim.mode not in ("ground",) and time.time() < end:
        time.sleep(0.1)
    time.sleep(0.3)
    with lock:
        if not any(p[5] != "ground" for p in sim.track):
            if args.json:
                print("SUMMARY " + json.dumps({"flown": False, "events": _key_events(flightlog.LOG.path)}))
            raise SystemExit("nothing was flown (see the AI messages above)")
        path = args.out or os.path.join(MEDIA_DIR, time.strftime("sim_%Y%m%d_%H%M%S.png"))
        render(sim, events, f"{llm.model}: {command}", path, obstacles, camera.collisions)
        print(f"\nfinal: {sim.pose()}  highest {max(p[3] for p in sim.track):.2f} m  "
              f"collisions {len(camera.collisions)}\npicture: {os.path.relpath(path)}", flush=True)
        flightlog.event("sim_end", pose=sim.pose(), collisions=len(camera.collisions))
        if args.json:
            flying = [p for p in sim.track if p[5] != "ground"]
            print("SUMMARY " + json.dumps({
                "flown": True, "x": round(sim.x, 3), "y": round(sim.y, 3), "z": round(sim.z, 3),
                "heading": round(sim.heading, 1), "mode": sim.mode,
                "end_distance": round(math.hypot(sim.x, sim.y), 3),
                "max_distance": round(max(math.hypot(p[1], p[2]) for p in sim.track), 3),
                "airborne_s": round(flying[-1][0] - flying[0][0], 2) if flying else 0,
                "collisions": len(camera.collisions), "auto_land": sup.landing,
                "axes": {a: {"on": ax.enabled, "flipped": ax.flipped, "gain": round(ax.gain, 1),
                             "min": round(ax.min, 1), "cap": ax.cap}
                         for a, ax in sup.axes.items()},
                "trim": {k: round(v, 2) for k, v in loop.pilot.trims().items()},
                "events": _key_events(flightlog.LOG.path)}), flush=True)


if __name__ == "__main__":
    main()
