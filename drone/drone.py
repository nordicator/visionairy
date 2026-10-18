#!/usr/bin/env python3
"""Fly a Sky Rider Cruise / cooingdv Wi-Fi toy drone from its video window.

  python drone.py                  # join the drone's Wi-Fi first, then run this
  python drone.py probe            # video only, sends NO flight commands
  python drone.py --webcam         # test window on the laptop's webcam: no drone, no commands
  python drone.py --save frames/   # also save every video frame as JPEG

Keys (in the video window) – hold a movement key to push harder, let go to stop:
  W/S forward/back   A/D left/right   UP/DOWN (or R/F) up/down
  T take off   L land   SPACE motors off   C gyro calibrate (on the ground)
  SHIFT + W/A/S/D/R/F nudge trim   K auto-trim on/off   BACKSPACE reset trim
  H headless mode on/off   P photo   V start/stop recording   O object boxes on/off
  ? show/hide the key list   ESC quit
  Z depth view on/off   M depth calibration: camera square to a wall --cal-distance m away (1 m)
  CLICK the video to fly there, same height (see target.py)   RIGHT-CLICK or a movement key cancels
  G follow the latest throw path, steered by the iOS app's camera + LiDAR (see follow.py); G again cancels
  SPEED 1/2/3: click the buttons in the window (no key)
Photos and recordings go to the media/ folder next to this script.

Drift: while no movement keys are held the video is used to measure which way the drone is
sliding, and auto-trim offsets the sticks to cancel it (see drift.py). Every time you let go it
auto-calibrates for a few seconds with faster correction, then holds. Trims are saved
to trim.json and reused next time.

Objects: humans, chairs, tables, TVs, bags, boxes and walls (incl. windows / doors) are boxed, named and
tracked with an ID (a small YOLOX model, downloaded once into models/, plus shape rules); anything
else is only shown while it's in the way, as "obstacle". Anything in the flight path that is close or closing in fast turns red with an
OBSTACLE AHEAD warning (see obstacles.py). It only warns - it never steers the drone.

Needs: pip install opencv-python numpy
"""
import argparse
import ctypes
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque

import ai_pilot
import calibration
import flightlog
import supervisor
from protocol import (A_CONTROL_OFF, A_HEARTBEAT, A_IP, A_CTRL_PORT, A_RTSP, B_CMD_CONTROL_OFF,
                      B_CMD_START, B_HELLO, B_IP, B_PORT, BReceiver, CENTER, FLAG_GYRO_CAL, FLAG_HEADLESS, FLAG_LAND,
                      FLAG_STOP, FLAG_TAKEOFF, SPEED_RANGES, Sticks, T_ACK, T_CTLMSG, T_FRAG,
                      a_control_packet, a_parse_status, b_ctlmsg_reply, b_parse, control_frame)

try:
    import cv2
    import numpy as np
    import drift
    import follow
    import hud
    import obstacles
    import depth
    import target
except ImportError:
    cv2 = None

WINDOW = "drone"
HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(HERE, "media")
TRIM_FILE = os.path.join(HERE, "trim.json")
PULSE_SECONDS = {FLAG_TAKEOFF: 1.0, FLAG_LAND: 1.0, FLAG_STOP: 1.0, FLAG_GYRO_CAL: 2.0}
ACTION_KEYS = {ord("t"): FLAG_TAKEOFF, ord("l"): FLAG_LAND, ord(" "): FLAG_STOP, ord("c"): FLAG_GYRO_CAL}
AI_ACTIONS = {"takeoff": FLAG_TAKEOFF, "land": FLAG_LAND, "stop": FLAG_STOP, "gyro": FLAG_GYRO_CAL}
ACTION_NAMES = {FLAG_TAKEOFF: "take off", FLAG_LAND: "land", FLAG_STOP: "motors off",
                FLAG_GYRO_CAL: "gyro calibrate"}

# Drone-frame axes -> stick bytes. +X = right, +Y = forward, +Z = up.
AXES = {"x": "roll", "y": "pitch", "z": "throttle"}
RAMP_START = 20     # % stick the moment a key goes down
RAMP_RATE = 100     # % per second while held -> full stick after 0.8 s
TRIM_STEP = 2.0     # % stick per SHIFT+key press
TRIM_KEYS = {ord("W"): ("y", 1), ord("S"): ("y", -1), ord("D"): ("x", 1), ord("A"): ("x", -1),
             ord("R"): ("z", 1), ord("F"): ("z", -1)}      # SHIFT gives the capital letter

# key name -> (OpenCV waitKey code, Windows virtual-key code)
KEY_CODES = {c: (ord(c), ord(c.upper())) for c in "wasdrftlpvhckozm"}
KEY_CODES.update(space=(32, 0x20), esc=(27, 0x1B), up=(None, 0x26), down=(None, 0x28),
                 shift=(None, 0x10), back=(8, 0x08))

HELP = "W/S forward/back  A/D left/right  UP/DOWN up/down  T take off  L land  SPACE off  ESC quit"


def local_ip_towards(ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((ip, 9))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def detect_variant():
    if local_ip_towards(B_IP).startswith("192.168.169."):
        return "b"
    if local_ip_towards(A_IP).startswith("192.168.1."):
        return "a"
    return None


class Keyboard:
    """Which keys are held down in the video window.

    On Windows this reads the real key state, so press and release take effect at once.
    Elsewhere it falls back to OpenCV key events plus the keyboard's auto-repeat."""
    REPEAT_GAP = 0.55   # fallback: a key counts as held this long after its last event

    def __init__(self, title):
        self.title = title
        self.last = {}
        self.user32 = None
        if sys.platform == "win32":
            self.user32 = ctypes.windll.user32
            self.user32.GetAsyncKeyState.restype = ctypes.c_short

    def saw(self, code):
        self.last[code] = time.time()

    def poll(self):
        """(set of key names held down, whether the video window has focus)"""
        if self.user32:
            hwnd = self.user32.FindWindowW(None, self.title)
            if hwnd:
                if self.user32.GetForegroundWindow() != hwnd:
                    return set(), False                     # window not focused: hands off
                return {n for n, (_, vk) in KEY_CODES.items()
                        if self.user32.GetAsyncKeyState(vk) & 0x8000}, True
        now = time.time()
        return {n for n, (code, _) in KEY_CODES.items()
                if code is not None and now - self.last.get(code, 0) < self.REPEAT_GAP}, True


def axes_from(pressed):
    if "shift" in pressed:                      # SHIFT + key trims instead of moving
        return dict.fromkeys(AXES, 0)

    def d(pos, neg):
        return int(bool(pressed & pos)) - int(bool(pressed & neg))
    return {"x": d({"d"}, {"a"}), "y": d({"w"}, {"s"}), "z": d({"r", "up"}, {"f", "down"})}


class Pilot:
    """Stick state. Holding a direction ramps that axis up; letting go recenters it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.level = dict.fromkeys(AXES, 0.0)   # -100..100 % of stick range
        self.trim = dict.fromkeys(AXES, 0.0)    # % offset added to every stick, cancels drift
        self.speed = 1                          # app speed mode 1-3: right-stick range (SPEED_RANGES)
        self.steering = False                   # click-to-go / AI is driving: ignore the speed mode
        self.headless = False                   # drone's headless mode: sticks relative to the pilot
        self.yaw = 0.0                          # -100..100 %, only the AI pilot turns
        self.source = None                      # AI plan: now -> levels or None, read at send time
        self.hold = {"x": 0.0, "y": 0.0}        # position-hold correction (supervisor.py), % stick
        self.invert = {"x": False, "y": False, "yaw": False}   # axes this drone reverses (calibration.py)
        self._pulses = {}                       # flag -> expires

    def set_hold(self, hold):
        with self.lock:
            self.hold = dict(hold)

    def set_speed(self, speed):
        with self.lock:
            self.speed = speed

    def update(self, held, dt):
        with self.lock:
            self.steering = False
            self.yaw = 0.0
            for axis, d in held.items():
                v = self.level[axis]
                if d == 0:
                    v = 0.0
                elif v * d <= 0:                # just pressed, or reversed
                    v = d * RAMP_START
                else:
                    v = d * min(100.0, abs(v) + RAMP_RATE * dt)
                self.level[axis] = v

    def steer(self, levels):
        """Set the sticks directly (click-to-go, AI), -100..100 % per axis of the full stick range.
        An optional "yaw" entry turns."""
        levels = dict(levels)
        with self.lock:
            self.steering = True
            self.yaw = levels.pop("yaw", 0.0)
            self.level.update(levels)

    def levels(self):
        with self.lock:
            return dict(self.level)

    def yaw_level(self):
        with self.lock:
            return self.yaw

    def trims(self):
        with self.lock:
            return dict(self.trim)

    def adjust_trim(self, axis, delta):
        lim = drift.TRIM_LIMIT[axis]
        with self.lock:
            self.trim[axis] = max(-lim, min(lim, self.trim[axis] + delta))

    def set_trims(self, trims):
        with self.lock:
            self.trim.update(trims)

    def pulse(self, flag):
        with self.lock:
            self._pulses = {flag: time.time() + PULSE_SECONDS[flag]}   # newest action wins

    def action(self):
        """(active flag or 0, fraction of its pulse still to run)"""
        now = time.time()
        with self.lock:
            for f, exp in self._pulses.items():
                if exp > now:
                    return f, (exp - now) / PULSE_SECONDS[f]
        return 0, 0.0

    def sticks(self):
        s = Sticks()                            # throttle 128 = hold altitude
        # an AI plan is read here, when the packet goes out, so its timing doesn't depend on the
        # video loop keeping up
        plan = self.source(time.time()) if self.source else None
        with self.lock:
            for axis, name in AXES.items():
                # like the app, the speed mode only limits the right stick (roll/pitch) the pilot
                # moves; trim and the throttle stay on the full range
                if plan is not None:
                    v = plan[axis] + self.trim[axis]
                else:
                    reach = 127 if axis == "z" or self.steering else SPEED_RANGES[self.speed]
                    v = self.level[axis] * reach / 127 + self.trim[axis]
                v += self.hold.get(axis, 0.0)
                if self.invert.get(axis):
                    v = -v
                setattr(s, name, CENTER + round(max(-100.0, min(100.0, v)) / 100 * 127))
            yaw = self.yaw if plan is None else plan["yaw"]
            s.yaw = ai_pilot.yaw_byte(-yaw if self.invert.get("yaw") else yaw)
        # headless rides in every packet while it's on, like the app's toggle (protocol.FLAG_HEADLESS)
        s.flags = self.action()[0] | (FLAG_HEADLESS if self.headless else 0)
        return s


class Link:
    def __init__(self, variant, fly, save_dir, verbose, camera="forward", autotrim=True, objects=True,
                 model="tiny", depth="fast", cal_distance=1.0, server="127.0.0.1"):
        self.variant, self.fly, self.save_dir, self.verbose = variant, fly, save_dir, verbose
        self.server = server
        self.camera, self.autotrim, self.objects, self.model = camera, autotrim, objects, model
        self.depth, self.cal_distance = depth, cal_distance
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 0))
        self.sock.settimeout(0.005)
        self.pilot = Pilot()
        self.running = True
        self.latest_jpeg = None
        self.frames = 0
        self.quality = None      # Variant B JPEG quality of the latest frame
        self.wifi = None         # (ssid, signal %) on Windows
        self.rx = BReceiver()
        self.ai = None           # ai_pilot.Bridge when flying with --ai
        self.agent = None        # ai_pilot.Agent
        self.sup = None          # supervisor.Supervisor (obstacle guard + position hold)
        self.depth_trust = None  # supervisor.DepthTrust: how far to believe the camera distances
        self.obs = None          # the video loop's ObstacleTracker (objects, depth)
        self.flying = False      # taken off and not landed (mirrors the video loop's flight_start)
        self.cal = calibration.default()

    def log(self, *a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)

    def send(self, pkt, dest):
        try:
            self.sock.sendto(pkt, dest)
        except OSError:          # Wi-Fi dropped / drone off – keep retrying
            pass

    def recv(self, size):
        """One packet, or None. Windows reports ICMP 'port unreachable' as ConnectionResetError."""
        try:
            return self.sock.recvfrom(size)[0]
        except socket.timeout:
            return None
        except OSError:
            time.sleep(0.05)
            return None

    def wifi_monitor(self):
        """Windows: SSID and signal strength of the drone's Wi-Fi link, refreshed every 3 s."""
        while self.running:
            try:
                out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True,
                                     text=True, timeout=5,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
                links = []
                for block in re.split(r"\n\s*\n", out):
                    ssid = re.search(r"^\s*SSID\s*:\s*(.+)$", block, re.M)
                    sig = re.search(r"^\s*Signal\s*:\s*(\d+)\s*%", block, re.M)
                    if ssid:
                        links.append((ssid.group(1).strip(), int(sig.group(1)) if sig else None))
                drone = [l for l in links if re.match(r"(HD|FHD)", l[0], re.I)]
                self.wifi = (drone or links or [None])[0]
            except (OSError, subprocess.SubprocessError):
                self.wifi = None
            time.sleep(3)

    def depth_info(self):
        """What's known about the depth setup, for supervisor.DepthTrust."""
        obs = self.obs
        d = obs.depth if obs is not None else None
        if d is None:
            return {"available": False, "status": "depth is off (--depth off)" if obs is not None else "starting"}
        if d.status != "ready" or d.model is None:
            return {"available": False, "status": f"depth model {d.status} (needs onnxruntime + the model file)"}
        cal = d.model.cal or {}
        try:
            age = (time.time() - os.path.getmtime(depth.CAL_FILE)) / 86400
        except OSError:
            age = None
        return {"available": True, "lens_guessed": d.camera.guessed, "cal_samples": len(cal.get("samples", [])),
                "scale": cal.get("scale", 1.0), "cal_age_days": age}

    # ---------------- Variant A ----------------
    def run_a(self):
        dest = (A_IP, A_CTRL_PORT)
        last_hb = 0.0
        last_ctl = 0.0
        while self.running:
            now = time.time()
            if now - last_hb >= 1.0:
                self.send(A_HEARTBEAT, dest)
                last_hb = now
            if self.fly and now - last_ctl >= 0.05:
                pkt = a_control_packet(self.pilot.sticks())
                self.send(pkt, dest)
                if self.verbose:
                    self.log("TX", pkt.hex(" "))
                last_ctl = now
            data = self.recv(64)
            if data:
                st = a_parse_status(data)
                self.log("RX", st)
                if "photo_button_count" in st:
                    self.send(b"\x09\x01", dest)
                if "video_button_count" in st:
                    self.send(b"\x09\x02", dest)
        if self.fly:
            self.send(A_CONTROL_OFF, dest)

    def video_webcam(self, index):
        """Test mode: the laptop's webcam stands in for the drone's camera."""
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not cap.isOpened():
            self.log(f"couldn't open webcam {index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            self.frames += 1
            self.latest_jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        cap.release()

    def video_a(self):
        if cv2 is None:
            self.log("OpenCV not installed – open", A_RTSP, "in VLC to see video.")
            return
        cap = cv2.VideoCapture(A_RTSP)
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(A_RTSP)
                continue
            self.frames += 1
            self.latest_jpeg = cv2.imencode(".jpg", frame)[1].tobytes()

    # ---------------- Variant B ----------------
    def run_b(self):
        dest = (B_IP, B_PORT)
        rx = self.rx
        last_hello = last_ack = last_frag = last_ctl = 0.0
        started = False
        while self.running:
            now = time.time()
            if rx.connected and now - last_frag > 3.0:
                self.log("video stopped for 3 s – reconnecting")
                self.rx = rx = BReceiver()
                started = False
            if not rx.connected and now - last_hello >= 0.1:
                self.send(B_HELLO, dest)
                last_hello = now
            if rx.connected and self.fly and started and now - last_ctl >= 0.05:
                rx.set_command(control_frame(self.pilot.sticks()))
                last_ctl = now
            if rx.connected and now - last_ack >= 0.025:
                self.send(rx.build_ack(idle=now - last_frag > 0.019), dest)
                last_ack = now
            data = self.recv(2048)
            if data is None:
                continue
            t, msg = b_parse(data)
            if t == T_FRAG:
                last_frag = time.time()
                jpg = rx.on_frag(msg)
                if jpg:
                    self.frames += 1
                    self.latest_jpeg = jpg
                    self.quality = msg.quality
                    if not started:
                        self.log(f"video up: {msg.width}x{msg.height} q{msg.quality}")
                        rx.set_command(B_CMD_START)
                        started = True
                        last_ctl = time.time() + 0.15   # let 0x64 ride a few acks first
            elif t == T_CTLMSG:
                seq, payload = msg
                self.send(b_ctlmsg_reply(seq), dest)
                self.log("RX message", payload.hex(" "))
            elif t == T_ACK:
                if self.verbose:
                    self.log("drone acked command", msg)
            elif t is not None:
                self.log("RX type", t, data[:32].hex(" "))
        if self.fly and rx.connected:
            rx.set_command(B_CMD_CONTROL_OFF)
            for _ in range(5):
                self.send(rx.build_ack(idle=True), dest)
                time.sleep(0.03)

    # ---------------- UI ----------------
    def save_frame(self, jpg):
        with open(os.path.join(self.save_dir, f"frame_{self.frames:06d}.jpg"), "wb") as f:
            f.write(jpg)

    def ui(self):
        if cv2 is None:
            last_saved = 0
            while self.running:
                if self.latest_jpeg and self.save_dir and self.frames != last_saved:
                    last_saved = self.frames
                    self.save_frame(self.latest_jpeg)
                time.sleep(0.05)
            return
        kb = Keyboard(WINDOW)
        img, shown = None, 0
        frame_times = deque(maxlen=120)
        toasts = []                     # (message, expires)
        writer = rec_start = rec_size = None
        flight_start = None
        show_keys = False                # the key list (?), hidden by default
        show_depth = False
        last_t = time.time()
        est = calibration.Calibrated(drift.DriftEstimator(self.camera), self.cal)
        auto = drift.AutoTrim(self.pilot, self.autotrim)
        if self.sup:
            self.sup.autotrim = auto
        self.pilot.set_trims(drift.load_trim(TRIM_FILE))
        saved_trim = self.pilot.trims()
        last_input = last_drift_log = 0.0
        goto = target.ClickTarget(self.camera)
        tracker = follow.PathFollower(follow.ServerFeed(self.server, log=self.log)) if self.fly else None
        obs = obstacles.ObstacleTracker(self.camera, self.model, self.log, self.depth)
        self.obs = obs
        goto_stop = None                # (object id, stop when its cautious distance <= this m) for AI fly_to
        obs.enabled = self.objects
        clicks = []                     # (button, canvas x, canvas y) from the mouse callback

        def on_mouse(event, x, y, flags, param):
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                clicks.append((event, x, y))

        def toast(msg):
            toasts.append((msg, time.time() + 2.5))
            del toasts[:-3]
            self.log(msg)
            flightlog.event("toast", msg=msg)

        def stamp():
            return time.strftime("%Y%m%d_%H%M%S")

        last_status, last_tel = None, 0.0
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)
        focused = False
        quit_ui = False                 # ESC / window closed: leave the loop but keep the radio up,
        while self.running and not quit_ui:     # so main() can still land the drone
            now = time.time()
            if self.frames != shown and self.latest_jpeg:
                shown = self.frames
                frame_times.append(now)
                if self.save_dir:
                    self.save_frame(self.latest_jpeg)
                decoded = cv2.imdecode(np.frombuffer(self.latest_jpeg, np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    img = decoded
                    est.update(img, now)
                    goto.update(img, now)
                    obs.update(img, now)
                    if writer:
                        writer.write(img if img.shape[1::-1] == rec_size else cv2.resize(img, rec_size))

            was_focused = focused
            pressed, focused = kb.poll()
            held = axes_from(pressed)
            if any(held.values()) or "shift" in pressed:
                last_input = now
            age = now - frame_times[-1] if frame_times else None
            if age is None or age > 0.5:
                est.valid = False
            airborne = None if flight_start is None else now - flight_start
            ai = self.ai if self.fly else None
            if ai:
                for msg in ai.take_notes():
                    toast(msg)
                for req in ai.take_requests():
                    if req[0] == "action":
                        flag = AI_ACTIONS[req[1]]
                        if goto.active and flag in (FLAG_LAND, FLAG_STOP):
                            goto.cancel()
                        self.pilot.pulse(flag)
                        if flag == FLAG_TAKEOFF:
                            flight_start = flight_start or now
                        elif flag in (FLAG_LAND, FLAG_STOP):
                            flight_start = None
                        flightlog.event("action", what=ACTION_NAMES[flag], source="ai")
                        toast(f"AI: {ACTION_NAMES[flag]}")
                    elif req[0] == "trim":
                        self.pilot.set_trims(req[1])
                        if req[2] is not None:
                            auto.enabled = bool(req[2])
                    elif req[0] == "goto" and img is not None and flight_start is not None:
                        goto.set(req[1], req[2], now)
                        goto_stop = (req[3], req[4]) if len(req) > 4 and req[4] else None
                        toast("AI: flying to target")
                airborne = None if flight_start is None else now - flight_start
                if ai.driving():
                    if any(held.values()):
                        ai.abort("pilot took over")
                    elif age is None or age > 1.0:
                        ai.abort("video lost")
                objs = obs.objects() if obs.enabled and img is not None else None
                ahead = obs.ahead_info() if obs.enabled and img is not None and obs.depth else None
                self.pilot.set_hold(self.sup.update(now, airborne, any(held.values()) or "shift" in pressed,
                                                    goto.active, age, est.vel, est.valid, objs, ahead))
                self.depth_trust.update(now, ahead[0] if ahead else None, self.sup.hold_state == "holding")
                if goto.active and goto_stop:              # AI fly_to with a stop distance (camera depth)
                    target_obj = next((o for o in objs or [] if o["id"] == goto_stop[0]), None)
                    if target_obj and target_obj.get("dist_safe") is not None and target_obj["dist_safe"] <= goto_stop[1]:
                        goto.cancel("arrived")
                        flightlog.event("fly_to_stop", object=goto_stop[0], dist=target_obj["dist"],
                                        dist_safe=target_obj["dist_safe"], stop_at=goto_stop[1])
                        toast(f"AI: stopped ~{target_obj['dist']:.1f} m from #{goto_stop[0]} (camera estimate)")
                if not goto.active:
                    goto_stop = None
            for button, cx, cy in clicks:
                if not self.fly:
                    continue
                speed = hud.speed_button_at(cx, cy)
                if speed:                               # speed is set only with these buttons
                    if button == cv2.EVENT_LBUTTONDOWN and speed != self.pilot.speed:
                        self.pilot.set_speed(speed)
                        toast(f"speed {speed}")
                    continue
                if not was_focused:                     # the click that focuses the window
                    continue
                if button == cv2.EVENT_RBUTTONDOWN:
                    if goto.active:
                        goto.cancel()
                        toast("go-to cancelled")
                elif img is None:
                    toast("no video yet")
                elif airborne is None:
                    toast("take off first, then click")
                else:
                    spot = hud.to_frame(cx, cy, img.shape)
                    if spot:
                        goto.set(*spot, now)
                        toast("flying to target")
            clicks.clear()
            if self.fly:
                going = goto.state
                if tracker.active and (any(held.values()) or airborne is None):
                    tracker.cancel()
                    self.pilot.steer(dict.fromkeys(AXES, 0.0))
                    toast("follow cancelled")
                if tracker.active:
                    self.pilot.steer(tracker.command(now))
                    last_input = now
                    if not tracker.active:              # finished, or lost the drone: land it
                        self.pilot.steer(dict.fromkeys(AXES, 0.0))
                        self.pilot.pulse(FLAG_LAND)
                        flight_start = None
                        toast("path done – landing" if tracker.state == "done" else f"follow stopped: {tracker.why} – landing")
                elif goto.active and (any(held.values()) or airborne is None):
                    goto.cancel()
                    toast("go-to cancelled")
                elif goto.active and age is not None and age > 1.5:
                    goto.cancel("lost")
                ai_levels = ai.levels(now) if ai else None
                if goto.active:
                    self.pilot.steer(goto.command(now, age is not None and age < 0.5))
                    last_input = now                    # auto-trim waits, then recalibrates
                elif ai_levels is not None:
                    self.pilot.steer(ai_levels)
                    step = ai.progress(now)
                    if step is None or step["kind"] != "hover":
                        last_input = now                # auto-trim keeps learning in hover steps
                elif not tracker.active:
                    self.pilot.update(held, now - last_t)
                if going == "tracking" and goto.state in ("arrived", "lost", "timeout"):
                    self.pilot.steer(dict.fromkeys(AXES, 0.0))
                    toast({"arrived": "target reached", "lost": "lost the target – stopped",
                           "timeout": "couldn't reach target – stopped"}[goto.state])
                was = auto.state
                auto.update(est, now - last_t, airborne, now - last_input)
                if was == "calibrating" and auto.state in ("learning", "steady"):
                    t = self.pilot.trims()
                    toast(f"calibrated  trim X {t['x']:+.0f}  Y {t['y']:+.0f}  Z {t['z']:+.0f}")
                if auto.state in ("calibrating", "learning") and now - last_drift_log > 3.0:
                    last_drift_log = now
                    t = self.pilot.trims()
                    self.log(f"drift {drift.describe(est.vel)} "
                             f"(x {est.vel['x']:+.3f} y {est.vel['y']:+.3f} z {est.vel['z']:+.3f}) -> "
                             f"trim x {t['x']:+.1f} y {t['y']:+.1f} z {t['z']:+.1f}")
            last_t = now
            self.flying = flight_start is not None
            trims = self.pilot.trims()
            if any(abs(trims[a] - saved_trim[a]) > 0.5 for a in trims):
                drift.save_trim(TRIM_FILE, trims)
                saved_trim = trims
            if age is None:
                status = ("CONNECTING", hud.YELLOW)
            elif age > 1.0:
                status = ("SIGNAL LOST", hud.RED)
            else:
                status = ("LIVE", hud.GREEN)
            res = None
            if img is not None:
                res = f"{img.shape[1]}x{img.shape[0]}" + (f" q{self.quality}" if self.quality else "")
            toasts[:] = [t for t in toasts if t[1] > now]
            state = {
                "t": now, "fly": self.fly, "variant": self.variant.upper(), "waiting": img is None,
                "webcam": self.variant == "webcam",
                "status": status, "wifi": self.wifi, "res": res,
                "fps": sum(1 for t in frame_times if now - t < 2.0) / 2.0,
                "age_ms": None if age is None else int(age * 1000),
                "flight": None if flight_start is None else now - flight_start,
                "rec": None if writer is None else now - rec_start,
                "levels": self.pilot.levels(), "action": self.pilot.action(),
                "pressed": pressed, "focused": focused, "show_keys": show_keys,
                "headless": self.pilot.headless,
                "trim": trims, "drift": dict(est.vel), "drift_valid": est.valid,
                "drift_text": drift.describe(est.vel), "auto": auto.state,
                "goto": goto.norm() if goto.active and img is not None else None,
                "frame_shape": None if img is None else img.shape, "speed": self.pilot.speed,
                "objects": obs.objects() if obs.enabled and img is not None else None,
                "model": obs.model_status,
                "depth_status": obs.depth_status if obs.enabled else None,
                "ahead": obs.ahead_info() if obs.enabled and img is not None else (None, False),
                "depth_img": obs.depth_map.colour((img.shape[1], img.shape[0]))
                if show_depth and obs.depth_map is not None and img is not None else None,
                "toasts": ([tracker.status()] if tracker and tracker.active else []) + [m for m, _ in reversed(toasts)],
            }
            state["ai"] = None
            if ai:
                state["levels"]["yaw"] = self.pilot.yaw_level()
                depth_now = self.depth_trust.status(state.get("ahead") if obs.depth else None)
                ai.publish({**ai_pilot.summarize(state, goto.state, auto.enabled), **self.sup.status(), **depth_now})
                state["ai"] = self.sup.hud(now, self.agent)
                state["ai"]["depth"] = depth_now["depth"]
                hold = self.sup.hold
                state["levels"]["x"] += hold["x"]               # show the stick actually sent
                state["levels"]["y"] += hold["y"]
            if status[0] != last_status:
                flightlog.event("link", status=status[0], age_ms=state["age_ms"], frames=self.frames)
                last_status = status[0]
            if now - last_tel >= supervisor.TEL_PERIOD:
                last_tel = now
                s = self.pilot.sticks()
                flightlog.event("tel", airborne=state["flight"] is not None, flight_s=state["flight"],
                                link=status[0], age_ms=state["age_ms"], fps=state["fps"],
                                levels={k: round(v, 1) for k, v in state["levels"].items()},
                                sticks=[s.roll, s.pitch, s.throttle, s.yaw, s.flags],
                                trim={k: round(v, 1) for k, v in trims.items()}, auto=auto.state,
                                drift={k: round(v, 4) for k, v in est.vel.items()}, drift_valid=est.valid,
                                drift_points=est.points, goto=goto.state, keys=sorted(pressed),
                                objects=[{"id": o["id"], "label": o["label"], "box": [round(v, 3) for v in o["box"]],
                                          "ttc": None if o["ttc"] is None else round(o["ttc"], 2),
                                          "threat": o["threat"]} for o in state["objects"] or []],
                                **(self.sup.tel_fields(now) if ai else {}))
            canvas = hud.fit(img)
            hud.draw(canvas, state)
            cv2.imshow(WINDOW, canvas)

            k = cv2.waitKey(15)
            if k != -1:
                k &= 0xFF
                kb.saw(k)
            if k == 27:
                quit_ui = True
            elif self.fly and k in ACTION_KEYS:
                flag = ACTION_KEYS[k]
                if ai:
                    ai.abort(ACTION_NAMES[flag] + " key")
                flightlog.event("action", what=ACTION_NAMES[flag], source="key")
                self.pilot.pulse(flag)
                if flag == FLAG_TAKEOFF:
                    flight_start = flight_start or now
                elif flag in (FLAG_LAND, FLAG_STOP):
                    flight_start = None
                self.log(ACTION_NAMES[flag])
            elif self.fly and k in TRIM_KEYS and (kb.user32 is None or "shift" in pressed):
                axis, d = TRIM_KEYS[k]
                self.pilot.adjust_trim(axis, d * TRIM_STEP)
                t = self.pilot.trims()
                toast(f"trim  X {t['x']:+.0f}  Y {t['y']:+.0f}  Z {t['z']:+.0f}")
            elif self.fly and k == ord("g"):
                if tracker.active:
                    tracker.cancel()
                    self.pilot.steer(dict.fromkeys(AXES, 0.0))
                    toast("follow cancelled")
                elif flight_start is None:
                    toast("take off first, then press G")
                else:
                    err = tracker.start(now)
                    if err:
                        toast(f"can't follow: {err}")
                    else:
                        goto.cancel()
                        toast("following the throw path – any movement key cancels")
            elif self.fly and k == ord("k"):
                auto.enabled = not auto.enabled
                toast("auto-trim " + ("on" if auto.enabled else "off"))
            elif self.fly and k == 8:
                self.pilot.set_trims(dict.fromkeys(AXES, 0.0))
                toast("trim reset")
            elif k == ord("p"):
                if self.latest_jpeg:
                    os.makedirs(MEDIA_DIR, exist_ok=True)
                    path = os.path.join(MEDIA_DIR, f"photo_{stamp()}.jpg")
                    with open(path, "wb") as f:
                        f.write(self.latest_jpeg)
                    toast(f"photo saved: media/{os.path.basename(path)}")
                else:
                    toast("no video yet")
            elif k == ord("v"):
                if writer:
                    writer.release()
                    writer = None
                    toast("recording saved")
                elif img is not None:
                    os.makedirs(MEDIA_DIR, exist_ok=True)
                    path = os.path.join(MEDIA_DIR, f"video_{stamp()}.avi")
                    fps = max(5.0, state["fps"]) if state["fps"] else 20.0
                    rec_size = img.shape[1::-1]
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, rec_size)
                    rec_start = now
                    toast(f"recording: media/{os.path.basename(path)}")
                else:
                    toast("no video yet")
            elif self.fly and k == ord("h"):
                # headless: the drone takes "forward" as the way its nose pointed when this was
                # switched on, whichever way it turns afterwards - point it away from you first
                self.pilot.headless = not self.pilot.headless
                toast("headless on - forward = away from you" if self.pilot.headless else "headless off")
                flightlog.event("headless", on=self.pilot.headless)
                if ai and self.pilot.headless:
                    ai.abort("headless mode switched on")   # its plans assume forward = the nose
            elif k == ord("?") or k == ord("/"):
                show_keys = not show_keys
            elif k == ord("z"):
                show_depth = not show_depth
                toast("depth view " + ("on" if show_depth else "off"))
            elif k == ord("m"):
                if img is None or obs.depth is None or obs.depth.status != "ready":
                    toast("depth not ready")
                else:
                    scale = obs.calibrate_depth(img, self.cal_distance)
                    toast(f"depth calibrated at {self.cal_distance:.2f} m: scale x{scale:.2f}")
            elif k == ord("o"):
                obs.enabled = not obs.enabled
                toast("object tracking " + ("on" if obs.enabled else "off"))
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # closed with the X
                quit_ui = True
        if writer:
            writer.release()
            self.log("recording saved")
        drift.save_trim(TRIM_FILE, self.pilot.trims())
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="fly", choices=["fly", "probe"])
    ap.add_argument("--variant", choices=["a", "b"], help="force hardware variant (default: auto)")
    ap.add_argument("--save", metavar="DIR", help="save every video frame as JPEG")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every packet sent")
    ap.add_argument("--camera", choices=["forward", "down"], default="forward",
                    help="which way the drone's camera points (for drift detection)")
    ap.add_argument("--no-autotrim", action="store_true", help="start with auto-trim off (K toggles)")
    ap.add_argument("--no-objects", action="store_true", help="start with object tracking off (O toggles)")
    ap.add_argument("--depth", choices=["fast", "fine", "off"], default="fast",
                    help="distance model: fast (default, ~50 ms on a GPU), fine (sharper, slow), off")
    ap.add_argument("--cal-distance", type=float, default=1.0, metavar="M",
                    help="for depth calibration (key M): metres from the camera to the wall (default 1)")
    ap.add_argument("--model", choices=["tiny", "nano", "off"], default="tiny",
                    help="object-naming model: tiny (default), nano (faster CPUs), off (unnamed boxes)")
    ap.add_argument("--webcam", nargs="?", const=0, type=int, metavar="N",
                    help="test window on webcam N (default 0) instead of the drone: no flight commands")
    ap.add_argument("--hfov", type=float, metavar="DEG",
                    help="camera field of view for depth, until calibrate_camera.py has measured it "
                         "(default: 90 for the drone, 65 for a webcam)")
    ap.add_argument("--server", default="127.0.0.1",
                    help="laptop running server.js, for G (follow a throw path); default this laptop")
    ap.add_argument("--no-calibration", action="store_true",
                    help="ignore calibration.json (what preflight.py --fly measured)")
    ap.add_argument("--first-flight", action="store_true",
                    help="gentle settings for a first test: AI sticks 40%%, safe area 1.5 m, ceiling 1.6 m, "
                         "flight limit 3 min, softer hold")
    ap.add_argument("--ai", action="store_true",
                    help="also take typed commands for an LLM pilot in this terminal (see ai_pilot.py)")
    ap.add_argument("--ai-model", help="override SPAR_MODEL for --ai")
    ap.add_argument("--ai-max-stick", type=float, default=60.0, help="AI stick cap in %% (default 60)")
    ap.add_argument("--no-ai-vision", action="store_true", help="don't send camera frames to the AI")
    ap.add_argument("--no-ai-guard", action="store_true", help="turn off the obstacle guard (stops AI plans)")
    ap.add_argument("--no-ai-hold", action="store_true", help="turn off position hold while the AI hovers")
    ap.add_argument("--ai-hold-gain", type=float, default=supervisor.HOLD_GAIN,
                    help=f"hold: %% stick per frame-width/s of slide (default {supervisor.HOLD_GAIN:g})")
    ap.add_argument("--ai-hold-max", type=float, default=supervisor.HOLD_MAX,
                    help=f"hold: max correction in %% stick (default {supervisor.HOLD_MAX:g})")
    ap.add_argument("--no-ai-gyro", action="store_true",
                    help="don't calibrate the gyro automatically before the AI's first take off")
    ap.add_argument("--ai-max-distance", type=float, default=3.0,
                    help="refuse AI plans that go further than this from the take-off spot, m (0 = off)")
    ap.add_argument("--ai-ceiling", type=float, default=2.0, help="refuse AI plans that climb above this, m (0 = off)")
    ap.add_argument("--ai-lost-land", type=float, default=supervisor.LOST_LAND_S,
                    help=f"land after this many s without video while flying (0 = never, default "
                         f"{supervisor.LOST_LAND_S:g})")
    ap.add_argument("--ai-max-flight", type=float, default=supervisor.FLIGHT_MAX_S,
                    help=f"land after this many s of flight, battery (0 = never, default {supervisor.FLIGHT_MAX_S:g})")
    ap.add_argument("--ai-hold-min", type=float, default=supervisor.HOLD_MIN,
                    help=f"hold: smallest push in %% stick, below which the drone doesn't react "
                         f"(default {supervisor.HOLD_MIN:g})")
    args = ap.parse_args()
    if args.first_flight:
        args.ai_max_stick = min(args.ai_max_stick, 40.0)
        args.ai_max_distance = min(args.ai_max_distance or 1.5, 1.5)
        args.ai_ceiling = min(args.ai_ceiling or 1.6, 1.6)
        args.ai_max_flight = min(args.ai_max_flight or 180.0, 180.0)
        args.ai_hold_gain = min(args.ai_hold_gain, 80.0)

    if args.webcam is not None and cv2 is None:
        sys.exit("The webcam test needs OpenCV: pip install opencv-python numpy")
    if args.webcam is not None:
        # the webcam isn't the drone's camera: its own lens guess and calibration files
        depth.CAMERA_FILE = os.path.join(HERE, "camera_webcam.json")
        depth.CAL_FILE = os.path.join(HERE, "depth_cal_webcam.json")
        depth.CAMERA_HFOV = args.hfov or 65.0
        args.mode = "probe"
    elif args.hfov and cv2 is not None:
        depth.CAMERA_HFOV = args.hfov
    variant = "webcam" if args.webcam is not None else args.variant or detect_variant()
    if not variant:
        sys.exit("Not on a drone network (expected 192.168.1.x or 192.168.169.x). "
                 "Join the drone's Wi-Fi or pass --variant.")
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    fly = args.mode == "fly"
    if fly and cv2 is None:
        sys.exit("Flying needs OpenCV for the video window: pip install opencv-python numpy")

    link = Link(variant, fly, args.save, args.verbose, args.camera, not args.no_autotrim, not args.no_objects,
                None if args.model == "off" else args.model, None if args.depth == "off" else args.depth,
                args.cal_distance, args.server)
    if not args.no_calibration:
        link.cal.update(calibration.load())
        link.pilot.invert = dict(link.cal["invert"])
    log_path = flightlog.LOG.open("flight")
    flightlog.event("session", argv=sys.argv, args=vars(args), variant=variant, platform=sys.platform,
                    python=sys.version.split()[0], opencv=getattr(cv2, "__version__", None))
    link.log(f"flight log: {os.path.relpath(log_path)}  (read it with: python flightlog.py)")
    flightlog.event("calibration", **link.cal)
    if link.cal["measured"]:
        link.log(f"calibration from {link.cal['measured']}: {calibration.describe(link.cal)}")
    elif variant != "webcam":
        link.log("no calibration yet: run  python preflight.py --fly  for a smoother flight")
    if variant == "webcam":
        link.log(f"webcam {args.webcam} test - no drone, no flight commands. ? shows the keys, ESC quits")
        threads = [threading.Thread(target=link.video_webcam, args=(args.webcam,), daemon=True)]
    else:
        link.log(f"variant {variant.upper()} – {'A: 192.168.1.1' if variant == 'a' else 'B: 192.168.169.1:8800'}")
        threads = [threading.Thread(target=link.run_a if variant == "a" else link.run_b, daemon=True)]
    if fly:
        link.log("click the video window, then:", HELP)
    if args.ai:
        if not fly:
            sys.exit("--ai needs fly mode (not probe or --webcam)")
        llm = ai_pilot.LLM(model=args.ai_model)
        llm.log = link.log
        link.ai = ai_pilot.Bridge(args.ai_max_stick)
        link.pilot.source = link.ai.levels
        link.sup = supervisor.Supervisor(link.ai, args.camera, not args.no_ai_guard, not args.no_ai_hold,
                                         args.ai_hold_gain, args.ai_hold_max, args.ai_hold_min,
                                         args.ai_lost_land, args.ai_max_flight)
        link.sup.pilot = link.pilot
        link.depth_trust = supervisor.DepthTrust(link.depth_info)
        link.agent = ai_pilot.Agent(llm, link.ai, lambda: link.latest_jpeg, not args.no_ai_vision, link.log,
                                    ai_pilot.Envelope(args.ai_max_distance, args.ai_ceiling),
                                    auto_gyro=not args.no_ai_gyro)
        ai_pilot.start_console(link.agent, link.ai, lambda word: link.ai.request("action", word))
        link.log(f"AI pilot: {llm.model or '(no SPAR_MODEL set)'} at {llm.base}, sticks capped at "
                 f"{args.ai_max_stick:g}%, guard {'off' if args.no_ai_guard else 'on'}, "
                 f"hold {'off' if args.no_ai_hold else 'on'}, vision {'off' if args.no_ai_vision else 'on'}")
    if variant == "a":
        threads.append(threading.Thread(target=link.video_a, daemon=True))
    if sys.platform == "win32" and variant != "webcam":
        threads.append(threading.Thread(target=link.wifi_monitor, daemon=True))
    for t in threads:
        t.start()
    crash = None
    try:
        link.ui()
    except KeyboardInterrupt:
        pass
    except Exception as e:                  # land first, then report the crash
        flightlog.error("ui", e)
        crash = e
    if fly and link.flying:
        land_before_exit(link, "program crashed" if crash else "quitting")
    link.running = False
    time.sleep(0.3)
    link.log(f"done, {link.frames} frames received")
    flightlog.event("session_end", frames=link.frames)
    if crash:
        raise crash


EXIT_LAND_WAIT = 3.5    # s the radio keeps running after a landing command on the way out


def land_before_exit(link, why):
    """Quitting (ESC, closing the window, Ctrl-C) or crashing mid-air: land before the radio stops."""
    link.log(f"still flying while {why}: landing first (Ctrl-C again to skip)")
    flightlog.event("exit_land", why=why)
    if link.ai:
        link.ai.abort(f"program {why}")
    link.pilot.source = None
    link.pilot.set_hold({"x": 0.0, "y": 0.0})
    link.pilot.update(dict.fromkeys(AXES, 0), 0.0)
    link.pilot.pulse(FLAG_LAND)
    try:
        time.sleep(EXIT_LAND_WAIT)
    except KeyboardInterrupt:
        link.log("skipped the landing wait")


if __name__ == "__main__":
    main()
