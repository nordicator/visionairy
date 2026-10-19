"""Dress rehearsal of the real launch: `python drone.py --ai` with default settings, end to end.

drone.main() runs for real (argument parsing, calibration, flight log, supervisor, agent, the ai> prompt,
the video loop, the radio thread) against a fake Variant A drone on this machine: video comes from a
local file instead of RTSP and control packets go to a local UDP socket, where they're checked byte by
byte. A command is typed into the prompt, a scripted LLM plans take off -> hover -> forward -> hover, and
ESC is pressed while it's still airborne, which must land it before the program exits. No window, no
network, no drone, no LLM quota.
"""
import io
import json
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np

import ai_pilot
import calibration
import drift
import drone
import protocol

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "media", "launch_test.avi")
RUN_S = 16.0            # ESC is pressed after this long

# ---- a textured video standing in for the drone's RTSP stream ------------------------
os.makedirs(os.path.dirname(VIDEO), exist_ok=True)
rng = np.random.default_rng(3)
scene = cv2.GaussianBlur((rng.random((480, 900, 3)) * 255).astype(np.uint8), (7, 7), 0)
w = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 360))
for i in range(75):
    w.write(scene[40:400, 60 + i:700 + i])
w.release()

# ---- the fake drone's control port ---------------------------------------------------
srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
srv.bind(("127.0.0.1", 0))
srv.settimeout(0.05)
packets = []                                 # (time, bytes)
listening = True


def listen():
    while listening:
        try:
            d, _ = srv.recvfrom(64)
            packets.append((time.time(), d))
        except socket.timeout:
            pass


threading.Thread(target=listen, daemon=True).start()
drone.A_IP, drone.A_CTRL_PORT, drone.A_RTSP = "127.0.0.1", srv.getsockname()[1], VIDEO
drone.EXIT_LAND_WAIT = 1.5


# ---- no window; ESC after RUN_S ------------------------------------------------------
start = time.time()
cv2.namedWindow = lambda *a, **k: None
cv2.setMouseCallback = lambda *a, **k: None
cv2.imshow = lambda *a: None
cv2.destroyAllWindows = lambda: None
cv2.getWindowProperty = lambda *a: 1.0
cv2.waitKey = lambda ms=0: (time.sleep(ms / 1000), 27 if time.time() - start > RUN_S else -1)[1]

# ---- keep the real trim / calibration files out of it ---------------------------------
drift.load_trim = lambda path: {}
drift.save_trim = lambda path, trim: None
calibration.load = lambda path=None: calibration.default()


class ScriptLLM:
    """The LLM, scripted: one plan, then a reply."""
    chain, model, base, last_model = ["scripted"], "scripted", "local", "scripted"
    log = staticmethod(print)

    def __init__(self, model=None):
        self.turns = [[("sequence", {"steps": [{"action": "takeoff"}, {"duration_s": 3},
                                               {"forward": 25, "duration_s": 1.0}, {"duration_s": 20}]})]]

    def chat(self, messages, tools):
        if not self.turns:
            return {"content": "Done."}
        return {"content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
            "name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(self.turns.pop(0))]}


ai_pilot.LLM = ScriptLLM


class SlowStdin(io.StringIO):
    """Types the command a moment after start, like a person at the ai> prompt."""

    def readline(self, *a):
        time.sleep(2.0)
        return super().readline(*a)


sys.stdin = SlowStdin("take off, hover, then fly forward a bit\n")
sys.argv = ["drone.py", "--ai", "--variant", "a", "--model", "off"]
drone.main()
listening = False

# ---- what the drone received ---------------------------------------------------------
ctl = [(t - start, p[1:]) for t, p in packets if p[:1] == b"\x03" and len(p) == 9]
assert ctl, "no control packets"
assert any(p == b"\x01\x01" for _, p in packets), "no heartbeat"
for _, f in ctl:                                       # every frame well formed
    assert f[0] == 0x66 and f[7] == 0x99 and f[6] == f[1] ^ f[2] ^ f[3] ^ f[4] ^ f[5], f.hex()
flag_times = {name: [t for t, f in ctl if f[5] & bit] for name, bit in
              (("gyro", protocol.FLAG_GYRO_CAL), ("takeoff", protocol.FLAG_TAKEOFF), ("land", protocol.FLAG_LAND))}
fwd = [t for t, f in ctl if f[2] >= 155]     # forward 25% = byte 160, give or take the hold
rate = len(ctl) / (ctl[-1][0] - ctl[0][0])
print(f"control packets: {len(ctl)} at {rate:.0f}/s")
print("gyro calibrate at", round(flag_times["gyro"][0], 1) if flag_times["gyro"] else None,
      "| take off at", round(flag_times["takeoff"][0], 1) if flag_times["takeoff"] else None,
      "| forward from", round(fwd[0], 1) if fwd else None, "to", round(fwd[-1], 1) if fwd else None,
      "| land at", round(flag_times["land"][0], 1) if flag_times["land"] else None)
assert 15 <= rate <= 25, "control rate should be ~20/s"
assert flag_times["gyro"] and flag_times["takeoff"] and flag_times["gyro"][0] < flag_times["takeoff"][0], \
    "gyro calibration must come before take off"
assert fwd and 0.7 <= fwd[-1] - fwd[0] <= 1.3 and fwd[0] > flag_times["takeoff"][0] + 2.5, \
    "forward step should last ~1 s, after the hover"
assert flag_times["land"] and flag_times["land"][0] > RUN_S - 0.5, "ESC while airborne must land it"
assert packets[-1][1] in (b"\x08\x01",) or any(p == b"\x08\x01" for _, p in packets[-5:]), "control-off on exit"
log = max((os.path.join(HERE, "logs", f) for f in os.listdir(os.path.join(HERE, "logs")) if f.startswith("flight_")),
          key=os.path.getmtime)
evs = [json.loads(line)["ev"] for line in open(log)]
for e in ("session", "calibration", "command", "plan_start", "auto_gyro_calibrate", "action", "exit_land",
          "session_end"):
    assert e in evs, f"{e} missing from the flight log"
assert "error" not in evs
print("flight log complete:", os.path.relpath(log))
print("LAUNCH REHEARSAL PASSED")
