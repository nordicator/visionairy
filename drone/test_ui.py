"""Headless run of drone.py's real video loop with --ai wiring: no window, no drone, no network.

The OpenCV window calls are stubbed out, video frames are fed in, a scripted LLM flies a plan, and an
obstacle appears in the path partway through, so the guard, hold, HUD panel, telemetry and flight
log all run through the same code as a real flight.
"""
import json
import os
import threading
import time

import cv2
import numpy as np

import ai_pilot
import drone
import flightlog
import supervisor

# ---- no window: stub the GUI calls the loop makes --------------------------------
shown = []
cv2.namedWindow = lambda *a, **k: None
cv2.setMouseCallback = lambda *a, **k: None
cv2.imshow = lambda name, img: shown.append(img)
cv2.waitKey = lambda ms=0: (time.sleep(ms / 1000), -1)[1]
cv2.getWindowProperty = lambda *a: 1.0
cv2.destroyAllWindows = lambda: None


class FakeLLM:
    model, base, chain, last_model, log = "fake", "local", ["fake"], "fake", print

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools):
        if not self.script:
            return {"content": "done"}
        calls = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                 for i, (n, a) in enumerate(self.script.pop(0))]
        return {"content": None, "tool_calls": calls}


log_path = flightlog.LOG.open("test_ui")
link = drone.Link("b", True, None, False, objects=True, model=None)
link.pilot.trim = dict.fromkeys(drone.AXES, 0.0)
drone.drift.load_trim = lambda path: {}                 # don't read or write the real trim.json
drone.drift.save_trim = lambda path, trim: None
link.ai = ai_pilot.Bridge(60)
link.pilot.source = link.ai.levels
link.sup = supervisor.Supervisor(link.ai)
llm = FakeLLM([[("sequence", {"steps": [{"action": "takeoff"}, {"duration_s": 2.5},
                                        {"forward": 30, "duration_s": 4.0}, {"action": "land"}]})]])
link.agent = ai_pilot.Agent(llm, link.ai, lambda: link.latest_jpeg, True, lambda *a: None, auto_gyro=False)

# ---- video: a textured frame, then an obstacle box filling the middle --------------
rng = np.random.default_rng(1)
base = (rng.random((360, 640, 3)) * 255).astype(np.uint8)
base = cv2.GaussianBlur(base, (5, 5), 0)
obstacle_from = time.time() + 6.0


def feed():
    i = 0
    while link.running:
        img = base.copy()
        if time.time() > obstacle_from:
            cv2.rectangle(img, (190, 40), (450, 330), (40, 40, 200), -1)
        link.latest_jpeg = cv2.imencode(".jpg", img)[1].tobytes()
        link.frames += 1
        i += 1
        time.sleep(0.04)


threading.Thread(target=feed, daemon=True).start()

# force the obstacle tracker to report a threat once the box is in view (its detector needs a real scene)
orig = drone.obstacles.ObstacleTracker.objects


def objects(self):
    if time.time() > obstacle_from:
        return [{"id": 3, "label": "box", "box": (0.3, 0.1, 0.7, 0.9), "ttc": 1.2, "threat": True, "lost": False}]
    return orig(self)


drone.obstacles.ObstacleTracker.objects = objects

ui = threading.Thread(target=link.ui, daemon=True)
ui.start()
while len(shown) < 10:                          # wait for live video, like a pilot would
    time.sleep(0.05)
obstacle_from = time.time() + 6.0
link.agent.submit("take off, hover, fly forward, land")
deadline = time.time() + 20
while time.time() < deadline and (link.agent.busy or time.time() < obstacle_from + 2):
    time.sleep(0.1)
link.running = False
ui.join(5)

recs = [json.loads(line) for line in open(log_path)]
evs = [r["ev"] for r in recs]
guard = [r for r in recs if r["ev"] == "guard_stop"]
tel = [r for r in recs if r["ev"] == "tel"]
holding = [r for r in tel if r.get("hold_state") == "holding"]
print(f"frames shown {len(shown)}, log events {len(recs)}, telemetry {len(tel)}")
assert shown and shown[-1].shape == (drone.hud.H, drone.hud.W, 3)
assert "plan_start" in evs and "plan_step" in evs and "action" in evs and "tool" in evs, set(evs)
assert guard and "box #3" in guard[0]["why"], guard
assert any(r["ev"] == "plan_end" and "obstacle" in (r["result"] or "") for r in recs)
assert tel and all("sticks" in r and "plan" in r for r in tel)
assert "error" not in evs, [r for r in recs if r["ev"] == "error"]
print("guard stopped the plan:", guard[0]["why"])
print("hold active in", len(holding), "telemetry samples")
cv2.imwrite(os.path.join("media", "hud_live_test.png"), shown[-1])
print("last HUD frame -> media/hud_live_test.png   log ->", os.path.relpath(log_path))

# quitting (ESC / window closed / Ctrl-C / crash) while still airborne lands first
assert link.flying, "the drone should still be airborne after the guard stop"
drone.EXIT_LAND_WAIT = 0.1
drone.land_before_exit(link, "quitting")
flag, _ = link.pilot.action()
assert flag == drone.FLAG_LAND, flag
assert any(json.loads(line)["ev"] == "exit_land" for line in open(log_path))
print("land before exit OK")
print("UI TEST PASSED")
