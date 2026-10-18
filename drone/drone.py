#!/usr/bin/env python3
"""Fly a Sky Rider Cruise / cooingdv Wi-Fi toy drone from its video window.

  python drone.py                  # join the drone's Wi-Fi first, then run this
  python drone.py probe            # video only, sends NO flight commands
  python drone.py --save frames/   # also save every video frame as JPEG

Keys (in the video window) – hold a movement key to push harder, let go to stop:
  W/S forward/back   A/D left/right   UP/DOWN (or R/F) up/down
  T take off   L land   SPACE motors off   C gyro calibrate (on the ground)
  SHIFT + W/A/S/D/R/F nudge trim   K auto-trim on/off   BACKSPACE reset trim
  P photo   V start/stop recording   H hide key map   ESC quit
Photos and recordings go to the media/ folder next to this script.

Drift: while no movement keys are held the video is used to measure which way the drone is
sliding, and auto-trim slowly offsets the sticks to cancel it (see drift.py). Trims are saved
to trim.json and reused next time.

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

from protocol import (A_CONTROL_OFF, A_HEARTBEAT, A_IP, A_CTRL_PORT, A_RTSP, B_CMD_CONTROL_OFF,
                      B_CMD_START, B_HELLO, B_IP, B_PORT, BReceiver, CENTER, FLAG_GYRO_CAL, FLAG_LAND,
                      FLAG_STOP, FLAG_TAKEOFF, Sticks, T_ACK, T_CTLMSG, T_FRAG, a_control_packet,
                      a_parse_status, b_ctlmsg_reply, b_parse, control_frame)

try:
    import cv2
    import numpy as np
    import drift
    import hud
except ImportError:
    cv2 = None

WINDOW = "drone"
HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(HERE, "media")
TRIM_FILE = os.path.join(HERE, "trim.json")
PULSE_SECONDS = {FLAG_TAKEOFF: 1.0, FLAG_LAND: 1.0, FLAG_STOP: 1.0, FLAG_GYRO_CAL: 2.0}
ACTION_KEYS = {ord("t"): FLAG_TAKEOFF, ord("l"): FLAG_LAND, ord(" "): FLAG_STOP, ord("c"): FLAG_GYRO_CAL}
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
KEY_CODES = {c: (ord(c), ord(c.upper())) for c in "wasdrftlpvhck"}
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
        self._pulses = {}                       # flag -> expires

    def update(self, held, dt):
        with self.lock:
            for axis, d in held.items():
                v = self.level[axis]
                if d == 0:
                    v = 0.0
                elif v * d <= 0:                # just pressed, or reversed
                    v = d * RAMP_START
                else:
                    v = d * min(100.0, abs(v) + RAMP_RATE * dt)
                self.level[axis] = v

    def levels(self):
        with self.lock:
            return dict(self.level)

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
        with self.lock:
            for axis, name in AXES.items():
                v = max(-100.0, min(100.0, self.level[axis] + self.trim[axis]))
                setattr(s, name, CENTER + round(v / 100 * 127))
        s.flags = self.action()[0]
        return s


class Link:
    def __init__(self, variant, fly, save_dir, verbose, camera="forward", autotrim=True):
        self.variant, self.fly, self.save_dir, self.verbose = variant, fly, save_dir, verbose
        self.camera, self.autotrim = camera, autotrim
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
        show_keys = True
        last_t = time.time()
        est = drift.DriftEstimator(self.camera)
        auto = drift.AutoTrim(self.pilot, self.autotrim)
        self.pilot.set_trims(drift.load_trim(TRIM_FILE))
        saved_trim = self.pilot.trims()
        last_input = last_drift_log = 0.0

        def toast(msg):
            toasts.append((msg, time.time() + 2.5))
            del toasts[:-3]
            self.log(msg)

        def stamp():
            return time.strftime("%Y%m%d_%H%M%S")

        cv2.namedWindow(WINDOW)
        while self.running:
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
                    if writer:
                        writer.write(img if img.shape[1::-1] == rec_size else cv2.resize(img, rec_size))

            pressed, focused = kb.poll()
            held = axes_from(pressed)
            if any(held.values()) or "shift" in pressed:
                last_input = now
            age = now - frame_times[-1] if frame_times else None
            if age is None or age > 0.5:
                est.valid = False
            if self.fly:
                self.pilot.update(held, now - last_t)
                airborne = None if flight_start is None else now - flight_start
                auto.update(est, now - last_t, airborne, now - last_input)
                if auto.state == "learning" and now - last_drift_log > 3.0:
                    last_drift_log = now
                    t = self.pilot.trims()
                    self.log(f"drift {drift.describe(est.vel)} "
                             f"(x {est.vel['x']:+.3f} y {est.vel['y']:+.3f} z {est.vel['z']:+.3f}) -> "
                             f"trim x {t['x']:+.1f} y {t['y']:+.1f} z {t['z']:+.1f}")
            last_t = now
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
                "status": status, "wifi": self.wifi, "res": res,
                "fps": sum(1 for t in frame_times if now - t < 2.0) / 2.0,
                "age_ms": None if age is None else int(age * 1000),
                "flight": None if flight_start is None else now - flight_start,
                "rec": None if writer is None else now - rec_start,
                "levels": self.pilot.levels(), "action": self.pilot.action(),
                "pressed": pressed, "focused": focused, "show_keys": show_keys,
                "trim": trims, "drift": dict(est.vel), "drift_valid": est.valid,
                "drift_text": drift.describe(est.vel), "auto": auto.state,
                "toasts": [m for m, _ in reversed(toasts)],
            }
            canvas = hud.fit(img)
            hud.draw(canvas, state)
            cv2.imshow(WINDOW, canvas)

            k = cv2.waitKey(15)
            if k != -1:
                k &= 0xFF
                kb.saw(k)
            if k == 27:
                self.running = False
            elif self.fly and k in ACTION_KEYS:
                flag = ACTION_KEYS[k]
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
            elif k == ord("h"):
                show_keys = not show_keys
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # closed with the X
                self.running = False
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
    args = ap.parse_args()

    variant = args.variant or detect_variant()
    if not variant:
        sys.exit("Not on a drone network (expected 192.168.1.x or 192.168.169.x). "
                 "Join the drone's Wi-Fi or pass --variant.")
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    fly = args.mode == "fly"
    if fly and cv2 is None:
        sys.exit("Flying needs OpenCV for the video window: pip install opencv-python numpy")

    link = Link(variant, fly, args.save, args.verbose, args.camera, not args.no_autotrim)
    link.log(f"variant {variant.upper()} – {'A: 192.168.1.1' if variant == 'a' else 'B: 192.168.169.1:8800'}")
    if fly:
        link.log("click the video window, then:", HELP)
    threads = [threading.Thread(target=link.run_a if variant == "a" else link.run_b, daemon=True)]
    if variant == "a":
        threads.append(threading.Thread(target=link.video_a, daemon=True))
    if sys.platform == "win32":
        threads.append(threading.Thread(target=link.wifi_monitor, daemon=True))
    for t in threads:
        t.start()
    try:
        link.ui()
    except KeyboardInterrupt:
        pass
    link.running = False
    time.sleep(0.3)
    link.log(f"done, {link.frames} frames received")


if __name__ == "__main__":
    main()
