"""Offline test of the AI pilot: a scripted fake LLM drives the dry-run loop, no network or drone."""
import json, os, threading, time
import ai_pilot
from ai_pilot import Agent, Bridge, DryRun, yaw_byte


class FakeLLM:
    model, base = "fake", "local"

    def __init__(self, script):
        self.script, self.seen = list(script), []

    def chat(self, messages, tools):
        self.seen.append(messages[-1])
        if not self.script:
            return {"content": "done"}
        calls = [{"id": f"c{i}", "type": "function",
                  "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(self.script.pop(0))]
        return {"content": None, "tool_calls": calls}


def run(script, max_stick=60.0, setup=None):
    bridge = Bridge(max_stick)
    sim = DryRun(bridge)
    threading.Thread(target=sim.loop, daemon=True).start()
    llm = FakeLLM(script)
    agent = Agent(llm, bridge, log=lambda *a: None, auto_gyro=False)
    if setup:
        setup(bridge, agent)
    agent.submit("go")
    time.sleep(0.2)
    while agent.busy:
        time.sleep(0.05)
    tools = [json.loads(m["content"]) for m in agent.messages if m["role"] == "tool"]
    return tools, sim, bridge, agent


ai_pilot.TAKEOFF_WAIT = ai_pilot.LAND_WAIT = 0.2

# yaw skips the drone's 104..152 dead band
assert yaw_byte(0) == 128 and yaw_byte(1) > 152 and yaw_byte(-1) < 104 and yaw_byte(100) == 255
print("yaw mapping OK")

# moving on the ground is refused, takeoff then a timed sequence flies for the right time
t, sim, *_ = run([[("move", {"forward": 30, "duration_s": 0.5})],
                  [("takeoff", {})],
                  [("sequence", {"steps": [{"forward": 40, "duration_s": 0.4, "ramp_s": 0.1},
                                           {"right": 90, "turn": -50, "duration_s": 0.3}]})],
                  [("land", {})]])
assert "takeoff first" in t[0]["error"], t[0]
assert t[1]["result"] == "took off"
assert t[2]["result"] == "done" and abs(t[2]["flown_s"] - 0.75) < 0.12, t[2]   # + half the first ramp
assert t[2]["clamped_to_max"] == ["right"], t[2]
assert t[3]["result"] == "landed" and t[3]["airborne"] is False
print("takeoff / sequence timing / clamp / land OK:", t[2]["flown_s"], "s")

# bad arguments come back as errors the LLM can read, not crashes
t, *_ = run([[("takeoff", {})], [("move", {"forward": 20, "duration_s": 9})],
             [("sequence", {"steps": [{"duration_s": 30}] * 2})], [("teleport", {})]])
assert "duration_s" in t[1]["error"] and "split" in t[2]["error"] and "unknown" in t[3]["error"], t
print("validation OK")


# a whole flight in one sequence: take off, move, land; moving on the ground is refused
t, *_ = run([[("sequence", {"steps": [{"forward": 30, "duration_s": 0.3}]})],
             [("sequence", {"steps": [{"action": "takeoff"}, {"right": 30, "duration_s": 0.3},
                                      {"action": "land"}]})],
             [("sequence", {"steps": [{"action": "land"}]})]])
assert "takeoff step" in t[0]["error"], t[0]
assert t[1]["result"] == "done" and t[1]["airborne"] is False, t[1]
assert "not flying" in t[2]["error"], t[2]
print("whole-flight sequence OK")

# ramps are centred on step boundaries: each step's stick x time is exact whatever the ramps
b = Bridge()
steps = [{"levels": {"x": 0, "y": 50, "z": 0, "yaw": 0}, "duration": 1.0, "ramp": 0.3},
         {"levels": {"x": 0, "y": 0, "z": 0, "yaw": -40}, "duration": 0.5, "ramp": 0.4},
         {"levels": {"x": 0, "y": 0, "z": 0, "yaw": 0}, "duration": 0.5, "ramp": 0.2}]
done = []
threading.Thread(target=lambda: done.append(b.run(steps)), daemon=True).start()
time.sleep(0.05)
start = b.plan[0]["start"] - 0.15
area = {"y": 0.0, "yaw": 0.0}
dt = 0.0005
tt = start
while tt < start + 2.5:
    lv = b.levels(tt)
    if lv is None:
        break
    for k in area:
        area[k] += lv[k] * dt
    tt += dt
assert abs(area["y"] - 50.0) < 0.2 and abs(area["yaw"] + 20.0) < 0.2, area
print("ramp compensation OK:", {k: round(v, 2) for k, v in area.items()})

# the pilot stopping it aborts the plan and locks out further movement for that command
def stop_soon(bridge, agent):
    threading.Timer(0.6, lambda: bridge.abort("pilot took over")).start()


t, sim, bridge, _ = run([[("takeoff", {})], [("move", {"forward": 30, "duration_s": 3})],
                         [("move", {"forward": 30, "duration_s": 1})]], setup=stop_soon)
assert t[1]["result"] == "aborted: pilot took over", t[1]
assert "took over" in t[2]["error"], t[2]
assert bridge.levels(time.time()) is None
print("override OK")

# ---- supervisor: obstacle guard -------------------------------------------------
import supervisor

Z = {"x": 0.0, "y": 0.0, "z": 0.0}
wall = {"id": 7, "label": "wall", "box": (0.2, 0.1, 0.8, 0.9), "ttc": 1.1, "threat": True, "lost": False}
side = {"id": 8, "label": "chair", "box": (0.0, 0.3, 0.15, 0.7), "ttc": 0.8, "threat": False, "lost": False}


def guard_case(levels, objects, age=0.05):
    b = Bridge()
    sup = supervisor.Supervisor(b, guard=True)
    out = []
    th = threading.Thread(target=lambda: out.append(b.run([{"levels": levels, "duration": 0.5, "ramp": 0.0}])),
                          daemon=True)
    th.start()
    time.sleep(0.05)
    while th.is_alive():                         # the video loop: guard + plan every frame
        sup.update(time.time(), 10.0, False, False, age, Z, True, objects)
        b.levels(time.time())
        time.sleep(0.02)
    return out[0], b, sup


fwd = {"x": 0.0, "y": 30.0, "z": 0.0, "yaw": 0.0}
res, b, sup = guard_case(fwd, [wall])
assert res.startswith("aborted: obstacle: wall #7") and sup.guard_state == "stopped", res
assert b.overrides == 0                         # a guard stop doesn't lock the AI out
assert b.ended["step"] == 1 and b.ended["what"] == "forward +30", b.ended
res, *_ = guard_case(fwd, [side])               # off to the side: keep going
assert res == "done", res
res, *_ = guard_case({"x": 30.0, "y": 0.0, "z": 0.0, "yaw": 0.0}, [wall])   # sliding sideways: not watched
assert res == "done", res
res, *_ = guard_case(fwd, [], age=0.9)          # flying forward blind
assert "can't see ahead" in res, res
print("guard OK")

# ---- supervisor: position hold ----------------------------------------------------
b = Bridge()
sup = supervisor.Supervisor(b, guard=True)
t = 1000.0
for i in range(40):                             # idle, sliding right at 0.05 frame-widths/s
    h = sup.update(t + i * 0.05, 10.0, False, False, 0.05, {"x": 0.05, "y": -0.2, "z": 0.0}, True, [])
assert h["x"] < -5.0 and abs(h["y"] - supervisor.HOLD_MAX) < 0.5, h      # push left; forward capped
assert sup.hold_state == "holding", sup.hold_state
h = sup.update(t + 3, 10.0, True, False, 0.05, {"x": 0.05, "y": 0.0, "z": 0.0}, True, [])
assert h == {"x": 0.0, "y": 0.0} and "pilot" in sup.hold_state         # keys: hands off
h = sup.update(t + 3.1, 10.0, False, False, 0.05, {"x": 0.05, "y": 0.0, "z": 0.0}, False, [])
assert h == {"x": 0.0, "y": 0.0} and "texture" in sup.hold_state
h = sup.update(t + 3.2, 0.5, False, False, 0.05, {"x": 0.05, "y": 0.0, "z": 0.0}, True, [])
assert h == {"x": 0.0, "y": 0.0} and "leaving the ground" in sup.hold_state
# turning: no hold during the turn and for HOLD_AFTER_TURN after it
b = Bridge()
sup = supervisor.Supervisor(b, guard=True)
steps = [{"levels": {"x": 0, "y": 0, "z": 0, "yaw": 50}, "duration": 0.3, "ramp": 0},
         {"levels": {"x": 0, "y": 0, "z": 0, "yaw": 0}, "duration": 3.0, "ramp": 0}]
threading.Thread(target=lambda: b.run(steps), daemon=True).start()
time.sleep(0.05)
vel = {"x": 0.08, "y": 0.0, "z": 0.0}
sup.update(time.time(), 10.0, False, False, 0.05, vel, True, [])
assert "plan moving" in sup.hold_state, sup.hold_state
time.sleep(0.4)
sup.update(time.time(), 10.0, False, False, 0.05, vel, True, [])
assert "after turn" in sup.hold_state, sup.hold_state
time.sleep(1.3)
for _ in range(10):
    sup.update(time.time(), 10.0, False, False, 0.05, vel, True, [])
    time.sleep(0.02)
assert sup.hold_state == "holding" and sup.hold["x"] < -3, (sup.hold_state, sup.hold)
b.abort("test", by_pilot=False)
print("hold OK")

# ---- hold softens itself when it wobbles ------------------------------------------
b = Bridge()
sup = supervisor.Supervisor(b, guard=True)
t = 2000.0
for i in range(200):                            # the slide flips direction every 0.3 s: a wobble
    v = 0.05 if (i // 6) % 2 == 0 else -0.05
    sup.update(t + i * 0.05, 10.0, False, False, 0.05, {"x": v, "y": 0.0, "z": 0.0}, True, [])
assert sup.axes["x"].gain < supervisor.HOLD_GAIN and sup.axes["x"].min < supervisor.HOLD_MIN, \
    (sup.axes["x"].gain, sup.axes["x"].min)
assert any(k.startswith("tune_x") for k in sup.alerts)
print("wobble self-tuning OK: gain", round(sup.axes["x"].gain), "min", round(sup.axes["x"].min, 1))

# ---- video watchdog + battery limit land the drone ----------------------------------
b = Bridge()
sup = supervisor.Supervisor(b, guard=True, lost_land_s=5, flight_max_s=60)
sup.update(3000.0, 10.0, False, False, 0.05, Z, True, [])
sup.update(3003.0, 13.0, False, False, 3.0, Z, False, [])
assert sup.landing is None and "video" in sup.alerts
sup.update(3006.0, 16.0, False, False, 6.0, Z, False, [])
assert "no video" in sup.landing and ("action", "land") in b.take_requests()
b = Bridge()
sup = supervisor.Supervisor(b, guard=True, flight_max_s=60)
sup.update(4000.0, 61.0, False, False, 0.05, Z, True, [])
assert "flight time" in sup.landing
print("watchdogs OK")

# ---- safe area: plans that leave the room are refused, with the reason ----------------
env = ai_pilot.Envelope(max_distance=3.0, ceiling=2.0)
mk = lambda y=0.0, z=0.0, d=1.0, action=None: {"levels": {"x": 0.0, "y": y, "z": z, "yaw": 0.0}, "duration": d,
                                             "ramp": 0.0, "action": action}
assert env.check([mk(action="takeoff", d=3), mk(y=50, d=2), mk(action="land", d=3)]) is None   # ~2 m: fine
assert "safe area" in env.check([mk(action="takeoff", d=3), mk(y=50, d=5)])                    # ~5 m
assert "ceiling" in env.check([mk(action="takeoff", d=3), mk(z=60, d=4)])                     # 1 + 2.4 m
assert "land step" in env.check([mk(action="takeoff", d=3), mk(z=-50, d=1.6), mk(d=1)])       # down to 0.2 m
env.commit([mk(action="takeoff", d=3), mk(y=50, d=2)])
assert abs(env.y - 2.0) < 0.05 and env.z == 1.0
assert "safe area" in env.check([mk(y=50, d=1.5)])                # 2 m + 1.5 m > 3 m
print("safe area OK")

# ---- the gyro is calibrated once, before the first take off of the session -----------
t, sim, bridge, agent = run([[("sequence", {"steps": [{"action": "takeoff"}, {"action": "land"}]})],
                             [("sequence", {"steps": [{"action": "takeoff"}, {"action": "land"}]})]],
                            setup=lambda b, a: setattr(a, "gyro_due", True))
gyros = sum(1 for e in json.loads(json.dumps([m for m in agent.messages if m["role"] == "tool"])))
assert agent.gyro_due is False and t[0]["result"] == "done" and t[1]["result"] == "done", t
print("auto gyro calibration OK")

# ---- camera depth: shown to the LLM with caution ------------------------------------
objs = [{"id": 4, "label": "chair", "box": (0.4, 0.3, 0.6, 0.8), "ttc": None, "threat": False, "lost": False,
         "dist": 1.84, "dist_safe": 1.47, "dist_fresh": True, "bearing": 3.2, "xyz": (0.1, -0.2, 1.84)}]
j = ai_pilot.objects_json(objs)[0]
assert j["distance_m"] == 1.84 and j["distance_m_cautious"] == 1.47 and j["bearing_deg"] == 3
assert j["position_m"] == {"right": 0.1, "up": 0.2, "forward": 1.84} and j["distance_is"] == "measured just now"
assert "distance_m" not in ai_pilot.objects_json([{**objs[0], "dist": None}])[0]
info = {"available": True, "lens_guessed": False, "cal_samples": 3, "scale": 1.1, "cal_age_days": 2}
dt = supervisor.DepthTrust(lambda: info)
assert dt.status((2.0, False))["depth"]["trust"] == "medium"
info["lens_guessed"] = True
st_ = dt.status((2.0, False))["depth"]
assert st_["trust"] == "low" and "lens" in st_["why"][0] and st_["ahead_m_cautious"] == 1.6
info["lens_guessed"] = False
for i in range(60):                                     # readings jumping 1.2 .. 2.4 m while hovering
    dt.update(5000 + i * 0.05, 1.2 if i % 2 else 2.4, True)
st_ = dt.status((1.8, False))["depth"]
assert st_["trust"] == "low" and any("jumps" in w for w in st_["why"]), st_
assert supervisor.DepthTrust(lambda: {"available": False, "status": "x"}).status(None)["depth"]["available"] is False
assert "agreed" in ai_pilot.depth_check(2.5, 1.5, (0, 0, 0), (0, 1.0, 0))
assert "DISAGREES" in ai_pilot.depth_check(2.5, 2.4, (0, 0, 0), (0, 1.0, 0))
assert ai_pilot.depth_check(2.5, 2.4, (0, 0, 0), (0, 1.0, 90)) is None      # turned: not comparable
print("depth status / trust / cross-check OK")


def depth_guard_case(ahead):
    b = Bridge()
    sup = supervisor.Supervisor(b, guard=True)
    out = []
    th = threading.Thread(target=lambda: out.append(b.run([{"levels": fwd, "duration": 0.8, "ramp": 0.0}])),
                          daemon=True)
    th.start()
    time.sleep(0.05)
    while th.is_alive():
        sup.update(time.time(), 10.0, False, False, 0.05, Z, True, [], ahead)
        b.levels(time.time())
        time.sleep(0.02)
    return out[0]


assert "camera depth" in depth_guard_case((0.9, True))
assert depth_guard_case((2.5, False)) == "done"
print("depth guard OK")

# headless mode on: the AI won't fly (forward would mean away from the pilot)
b = Bridge()
b.publish({"airborne": False, "headless_mode": True})
ag = Agent(FakeLLM([]), b, log=lambda *a: None, auto_gyro=False)
for tool, args in (("sequence", {"steps": [{"action": "takeoff"}, {"action": "land"}]}), ("takeoff", {})):
    assert "headless" in ag.execute(tool, args, [])["error"]
assert not b.take_requests()                            # nothing was sent to the drone
print("headless refusal OK")

# ---- HUD AI panel draws (saved for a look) ----------------------------------------
import hud
import cv2
import numpy as np


class _A:
    phase, phase_since, current_tool = "running", time.time() - 2, "sequence"
    last_command = "take off, fly a 1 metre square, turn 90 degrees right, then land"
    last_reply = "Flying a 1 m square at 30% with hovers between sides, then turning right."

    class llm:
        last_model, chain = "gemini-3.8-flash", ["gemini-3.8-flash"]


b = Bridge()
sup = supervisor.Supervisor(b, guard=True)
plan = [{"levels": {"x": 0, "y": 30, "z": 0, "yaw": 0}, "duration": 2, "ramp": 0.3},
        {"levels": {"x": 0, "y": 0, "z": 0, "yaw": 0}, "duration": 0.8, "ramp": 0}] * 4
threading.Thread(target=lambda: b.run(plan), daemon=True).start()
time.sleep(0.6)
sup.guard_state, sup.guard_detail = "watching", ""
st = {"t": time.time(), "waiting": False, "fly": True, "status": ("LIVE", hud.GREEN), "variant": "B",
      "wifi": None, "age_ms": 40, "res": "640x360", "fps": 25.0, "objects": [], "model": ("yolox tiny", True),
      "flight": 12.0, "rec": None, "goto": None, "speed": 1, "levels": {"x": 0, "y": 30, "z": 0},
      "trim": {"x": 1, "y": -2, "z": 0}, "drift": {"x": 0.01, "y": 0.0, "z": 0.0}, "drift_valid": True,
      "drift_text": "right", "auto": "paused", "focused": True, "pressed": set(), "show_keys": True,
      "action": (0, 0.0), "toasts": [], "frame_shape": (360, 640, 3), "ai": sup.hud(time.time(), _A)}
canvas = hud.fit(np.full((360, 640, 3), 90, np.uint8))
hud.draw(canvas, st)
os.makedirs("media", exist_ok=True)
cv2.imwrite("media/hud_ai_panel.png", canvas)
b.abort("test", by_pilot=False)
print("HUD AI panel OK -> media/hud_ai_panel.png")

print("all AI tests passed")
