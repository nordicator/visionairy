#!/usr/bin/env python3
"""LLM pilot: type what the drone should do, and an LLM turns it into exact, timed tool calls.

  python drone.py --ai                      # normal video window, plus an "ai>" prompt in this terminal
  python ai_pilot.py --list-models          # which models your key can use
  python ai_pilot.py --dry-run "take off, fly a 1 s square, land"   # no drone: prints the stick bytes

Works with any OpenAI-compatible chat-completions API (Spar IC, OpenAI, OpenRouter, vLLM, ...).
Put these in drone/.env (see .env.example) or the environment:

  SPAR_API_KEY      your key
  SPAR_BASE_URL     the API root, the part before /chat/completions (often ends in /v1)
  SPAR_MODEL        a model id from --list-models (it must support tool calling)
  SPAR_AUTH_HEADER  optional: header for the key if it isn't "Authorization: Bearer <key>"
  SPAR_TEMPERATURE  optional: leave unset if your model rejects it

Control: the LLM sets each stick as a % of its range (-100..100) for an exact time in seconds, with
optional linear ramps, and can chain up to 30 s of steps into one sequence that runs back to back
with no network delay between them. It can also take off, land, cut the motors, fly to a tracked
object, adjust trim, read the status (drift estimate, objects, obstacle warnings) and, with
--ai-vision, look at the camera frame. Anything in ai_notes.md is added to its instructions; put
your own measurements there ("forward 40% for 1 s = about 1.2 m").

Safety: sticks are capped at --ai-max-stick % (default 60). Any movement key, T/L/SPACE, losing
the video, or typing "stop" in the terminal stops what the AI is doing and lets the sticks
recenter. "land" and "kill" typed in the terminal go straight to the drone, skipping the LLM.
"""
import argparse
import base64
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque

import flightlog

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, ".env")
NOTES_FILE = os.path.join(HERE, "ai_notes.md")

AXES = {"right": "x", "forward": "y", "up": "z", "turn": "yaw"}   # tool argument -> stick level key
MAX_STEP = 5.0          # s, longest single stick step
MAX_PLAN = 45.0         # s, longest sequence (take off and landing included) / hover
MAX_ROUNDS = 40         # LLM replies per typed command
HISTORY = 60            # messages kept between commands
TAKEOFF_WAIT = 3.0      # s for the drone to climb out and settle
LAND_WAIT = 3.0
GYRO_WAIT = 2.5
GOTO_TIMEOUT = 25.0     # s, a little longer than target.TIMEOUT

DIRECT = {"stop", "land", "kill"}   # typed commands that skip the LLM


def load_env(path=ENV_FILE):
    """KEY=VALUE lines from drone/.env. Real environment variables win."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


class LLM:
    """Minimal OpenAI-compatible chat-completions client (standard library only)."""

    def __init__(self, base_url=None, key=None, model=None):
        load_env()
        self.base = (base_url or os.environ.get("SPAR_BASE_URL", "")).rstrip("/")
        if self.base.endswith("/chat/completions"):     # accept the full endpoint URL too
            self.base = self.base[:-len("/chat/completions")]
        self.key = key or os.environ.get("SPAR_API_KEY", "")
        self.model = model or os.environ.get("SPAR_MODEL", "")
        self.chain = [m.strip() for m in self.model.split(",") if m.strip()]   # first choice, then fallbacks
        self.no_effort = set()      # models that reject reasoning_effort
        self.last_model = None      # which model answered last
        self.log = print
        self.header = os.environ.get("SPAR_AUTH_HEADER", "Authorization")
        temp = os.environ.get("SPAR_TEMPERATURE")
        self.temperature = float(temp) if temp else None
        self.timeout = float(os.environ.get("SPAR_TIMEOUT", "30"))
        effort = os.environ.get("SPAR_REASONING_EFFORT", "low").strip().lower()
        self.effort = None if effort in ("", "default", "unset") else effort   # "none" is sent as is
        missing = [n for n, v in (("SPAR_API_KEY", self.key), ("SPAR_BASE_URL", self.base)) if not v]
        if missing:
            raise SystemExit(f"Set {' and '.join(missing)} in drone/.env (copy .env.example) or the environment.")

    def _call(self, method, path, body=None, retries=1):
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   self.header: f"Bearer {self.key}" if self.header.lower() == "authorization" else self.key}
        data = None if body is None else json.dumps(body).encode()
        for attempt in range(retries + 1):      # a stuck call leaves the drone hovering: don't wait forever
            last = attempt == retries
            req = urllib.request.Request(self.base + path, data, headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:600]
                if e.code >= 500 and not last:
                    continue
                raise RuntimeError(f"HTTP {e.code} from {self.base}{path}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if not last:
                    continue
                raise RuntimeError(f"can't reach {self.base}: {getattr(e, 'reason', e)}") from None

    def models(self):
        out = self._call("GET", "/models")
        items = out.get("data", []) if isinstance(out, dict) else out
        return sorted(m.get("id", "?") if isinstance(m, dict) else str(m) for m in items)

    def chat(self, messages, tools):
        """One reply, from the first model in SPAR_MODEL that answers (the rest are fallbacks)."""
        if not self.chain:
            raise RuntimeError("set SPAR_MODEL (python ai_pilot.py --list-models shows the choices)")
        for attempt in (1, 2):
            try:
                return self._chat_chain(messages, tools)
            except RateLimited as e:
                if attempt == 2:
                    raise RuntimeError(str(e)) from None
                self.log("AI: rate limited by every model - waiting 5 s")
                flightlog.event("llm_wait", why="rate limited", seconds=5)
                time.sleep(5)

    def _chat_chain(self, messages, tools):
        retries = 1 if len(self.chain) == 1 else 0     # with fallbacks, move on instead of retrying
        limited = 0
        for i, model in enumerate(self.chain):
            t0 = time.time()
            flightlog.event("llm_request", model=model, messages=len(messages),
                            last=_short(messages[-1].get("content") if messages else "", 300))
            try:
                out = self._chat_one(model, messages, tools, retries)
            except RuntimeError as e:
                flightlog.event("llm_error", model=model, seconds=round(time.time() - t0, 2), error=str(e)[:1000])
                limited += "HTTP 429" in str(e)
                if i == len(self.chain) - 1:
                    if limited == len(self.chain):
                        raise RateLimited(str(e)) from None
                    raise
                self.log(f"AI: {model} failed ({str(e)[:90]}) - trying {self.chain[i + 1]}")
                continue
            try:
                msg = out["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                flightlog.event("llm_error", model=model, error="unexpected reply", reply=out)
                raise RuntimeError(f"unexpected reply: {json.dumps(out)[:600]}") from None
            self.last_model = model
            flightlog.event("llm_response", model=model, seconds=round(time.time() - t0, 2),
                            finish=out["choices"][0].get("finish_reason"), usage=out.get("usage"),
                            content=msg.get("content"), tool_calls=msg.get("tool_calls"))
            return msg

    def _chat_one(self, model, messages, tools, retries):
        body = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.effort and model not in self.no_effort:
            body["reasoning_effort"] = self.effort
        try:
            return self._call("POST", "/chat/completions", body, retries)
        except RuntimeError as e:
            if not ("reasoning_effort" in body and "HTTP 400" in str(e)
                    and any(w in str(e).lower() for w in ("reasoning", "thinking"))):
                raise
            self.no_effort.add(model)               # this model doesn't take it: retry without
            body.pop("reasoning_effort")
            return self._call("POST", "/chat/completions", body, retries)


class RateLimited(RuntimeError):
    pass


class Bridge:
    """What the LLM thread and the video loop share.

    The video loop owns the drone: every frame it reads levels(), carries out take_requests() and
    publish()es the status. The LLM thread only schedules plans and requests and waits on them."""

    def __init__(self, max_stick=60.0):
        self.max_stick = max_stick
        self.cond = threading.Condition()
        self.plan = None        # [{start, end, ramp, levels, action, fired}], absolute times
        self.result = None      # how the last plan ended: "done" or "aborted: ..."
        self.ended = None       # progress() at the moment the last plan ended
        self.ended_t = 0.0
        self.step_now = None    # index of the step being flown, for logging transitions
        self.requests = deque()
        self.notes = deque()    # messages for the HUD
        self.status = {}
        self.overrides = 0      # times the AI was locked out of moving (keys, video lost, "stop", auto-landing)
        self.lock_reason = None

    # ---- video loop side
    def levels(self, now):
        """Stick levels (% per axis, plus yaw) the plan wants right now, or None when not driving."""
        with self.cond:
            if not self.plan:
                return None
            plan = self.plan
            for i, st in enumerate(plan):
                if st["action"] and not st["fired"] and now >= st["start"]:
                    st["fired"] = True
                    self.requests.append(("action", st["action"]))
                nxt = plan[i + 1] if i + 1 < len(plan) else None
                half = nxt["ramp"] / 2 if nxt else 0.0
                if now >= st["end"] - half:
                    continue
                if i != self.step_now:
                    self.step_now = i
                    flightlog.event("plan_step", step=i + 1, steps=len(plan), what=describe_step(st),
                                    levels=st["levels"], duration=round(st["end"] - st["start"], 3))
                # ramps are centred on the step boundaries, so a ramp never changes how far a step goes
                if st["ramp"] > 0 and now < st["start"] + st["ramp"] / 2:
                    prev = plan[i - 1]["levels"] if i else dict.fromkeys(st["levels"], 0.0)
                    return _blend(prev, st["levels"], (now - st["start"]) / st["ramp"] + 0.5)
                return dict(st["levels"])
            last = plan[-1]
            if now < last["end"]:      # can't happen: the last step has no successor ramp
                return dict(last["levels"])
            self.ended, self.ended_t = self._progress(now), now
            self.plan, self.result = None, "done"
            self.cond.notify_all()
            return None

    def _progress(self, now):
        plan = self.plan
        if not plan:
            return None
        i = next((k for k, st in enumerate(plan) if now < st["end"]), len(plan) - 1)
        st = plan[i]
        t0 = plan[0]["start"] - plan[0]["ramp"] / 2
        return {"step": i + 1, "steps": len(plan), "what": describe_step(st), "kind": step_kind(st),
                "levels": dict(st["levels"]), "left": round(max(0.0, st["end"] - now), 2),
                "elapsed": round(max(0.0, now - t0), 2), "total": round(plan[-1]["end"] - t0, 2),
                "kinds": [step_kind(p) for p in plan],
                "durations": [round(p["end"] - p["start"], 2) for p in plan]}

    def progress(self, now=None):
        """Where the running plan is (step, what it's doing, time left), or None. No side effects."""
        with self.cond:
            return self._progress(time.time() if now is None else now)

    def abort(self, why, by_pilot=True):
        with self.cond:
            if by_pilot:
                self.overrides += 1
                self.lock_reason = why
            if self.plan is None:
                return False
            self.ended, self.ended_t = self._progress(time.time()), time.time()
            self.plan, self.result = None, f"aborted: {why}"
            self.cond.notify_all()
        flightlog.event("abort", why=why, by_pilot=by_pilot, at=self.ended)
        self.note(f"AI stopped: {why}")
        return True

    def driving(self):
        with self.cond:
            return self.plan is not None

    def publish(self, status):
        with self.cond:
            self.status = status
            self.cond.notify_all()

    def take_requests(self):
        with self.cond:
            out = list(self.requests)
            self.requests.clear()
        return out

    def take_notes(self):
        with self.cond:
            out = list(self.notes)
            self.notes.clear()
        return out

    # ---- LLM thread side
    def request(self, *req):
        with self.cond:
            self.requests.append(req)

    def note(self, msg):
        with self.cond:
            self.notes.append(msg)

    def snapshot(self):
        with self.cond:
            return dict(self.status)

    def wait_until(self, pred, timeout):
        """Block until pred(status) is true; returns whether it became true in time."""
        end = time.time() + timeout
        with self.cond:
            while not pred(self.status):
                left = end - time.time()
                if left <= 0:
                    return False
                self.cond.wait(min(left, 0.1))
        return True

    def run(self, steps):
        """Fly the steps back to back from hover. Blocks until they finish or are aborted.

        Each step holds its levels for duration seconds. A step's ramp blends from the previous
        step's levels, centred on the boundary between them, so the total stick x time (and so the
        distance or angle) of every step is exactly level x duration whatever the ramps."""
        t = time.time() + steps[0]["ramp"] / 2
        plan = []
        for i, s in enumerate(steps):
            ramp = s["ramp"]
            if i:
                ramp = min(ramp, steps[i - 1]["duration"], s["duration"])
            plan.append({"start": t, "end": t + s["duration"], "ramp": min(ramp, s["duration"]),
                         "levels": s["levels"], "action": s.get("action"), "fired": False})
            t += s["duration"]
        flightlog.event("plan_start", steps=[{"what": describe_step(p), "levels": p["levels"],
                                              "duration": round(p["end"] - p["start"], 3), "ramp": p["ramp"],
                                              "action": p["action"]} for p in plan],
                        total=round(t - plan[0]["start"], 3))
        with self.cond:
            self.plan, self.result, self.ended, self.step_now = plan, None, None, None
            while self.plan is plan:
                if time.time() > t + 2.0:              # nobody is reading levels()
                    self.ended, self.ended_t = self._progress(time.time()), time.time()
                    self.plan, self.result = None, "aborted: video loop not running"
                    break
                self.cond.wait(0.1)
            result = self.result
        flightlog.event("plan_end", result=result, at=self.ended)
        return result


def step_kind(st):
    if st.get("action"):
        return st["action"]
    return "move" if any(abs(v) >= 0.5 for v in st["levels"].values()) else "hover"


def describe_step(st):
    """'forward 30 turn -50' / 'hover' / 'takeoff' for logs and the HUD."""
    kind = step_kind(st)
    if kind != "move":
        return kind
    names = {v: k for k, v in AXES.items()}
    return " ".join(f"{names[k]} {v:+.0f}" for k, v in st["levels"].items() if abs(v) >= 0.5)


def _blend(a, b, f):
    f = min(1.0, max(0.0, f))
    return {k: a[k] + (b[k] - a[k]) * f for k in b}


def yaw_byte(level):
    """Yaw % -> stick byte. The drone ignores yaw bytes 104-152, so any non-zero % starts past that."""
    if abs(level) < 0.5:
        return 128
    return 128 + round(math.copysign(25 + min(100.0, abs(level)) / 100 * 102, level))


def objects_json(objs):
    """Tracker objects (obstacles.py format) as plain JSON for the LLM: centre / size as picture fractions."""
    out = []
    for o in objs or []:
        x0, y0, x1, y1 = o["box"]
        j = {"id": o["id"], "label": o["label"],
             "x": round((x0 + x1) / 2, 3), "y": round((y0 + y1) / 2, 3),
             "width": round(x1 - x0, 3), "height": round(y1 - y0, 3),
             "seconds_to_contact": None if o["ttc"] is None else round(o["ttc"], 1),
             "in_path": bool(o["threat"]), "briefly_lost": bool(o["lost"])}
        if o.get("dist") is not None:                     # camera-based depth (depth.py): an estimate
            j["distance_m"] = round(o["dist"], 2)
            j["distance_m_cautious"] = round(o.get("dist_safe") or o["dist"] * 0.8, 2)
            j["distance_is"] = "measured just now" if o.get("dist_fresh") else "predicted, not re-measured lately"
            if o.get("bearing") is not None:
                j["bearing_deg"] = round(o["bearing"])
            if o.get("xyz"):
                x, y, z = o["xyz"]
                j["position_m"] = {"right": round(x, 2), "up": round(-y, 2), "forward": round(z, 2)}
        out.append(j)
    return out


def summarize(state, goto_state, auto_on):
    """The video loop's HUD state as plain JSON for the LLM (positions are 0..1 picture fractions)."""
    objects = objects_json(state.get("objects"))
    lv, tr, dv = state["levels"], state["trim"], state["drift"]
    return {
        "airborne": state["flight"] is not None,
        "flight_s": None if state["flight"] is None else round(state["flight"], 1),
        "link": state["status"][0], "video_age_ms": state["age_ms"], "fps": round(state["fps"], 1),
        "sticks_pct": {"right": round(lv["x"]), "forward": round(lv["y"]), "up": round(lv["z"]),
                       "turn": round(lv.get("yaw", 0))},
        "trim_pct": {"right": round(tr["x"], 1), "forward": round(tr["y"], 1), "up": round(tr["z"], 1)},
        "auto_trim": state["auto"] if auto_on else "off",
        "drift": state["drift_text"] if state["drift_valid"] else "unknown (low texture or no video)",
        "drift_frame_widths_per_s": {"right": round(dv["x"], 3), "forward": round(dv["y"], 3),
                                     "up": round(dv["z"], 3)},
        "fly_to": goto_state, "objects": objects,
        "obstacle_ahead": any(o["in_path"] for o in objects),
        "headless_mode": bool(state.get("headless")),
    }


def _num(desc, lo=None, hi=None):
    p = {"type": "number", "description": desc}
    if lo is not None:
        p["minimum"] = lo
    if hi is not None:
        p["maximum"] = hi
    return p


def _stick_props(max_stick):
    m = max_stick
    return {
        "right": _num(f"Roll %, + = slide right, - = left. -{m:g}..{m:g}. Default 0.", -m, m),
        "forward": _num(f"Pitch %, + = forward, - = back. -{m:g}..{m:g}. Default 0.", -m, m),
        "up": _num(f"Throttle %, + = climb, - = descend, 0 = hold altitude. -{m:g}..{m:g}. Default 0.", -m, m),
        "turn": _num(f"Yaw %, + = rotate clockwise (right), - = counter-clockwise. -{m:g}..{m:g}. Default 0.",
                     -m, m),
        "duration_s": _num(f"Exactly how long to hold these sticks, seconds (0.05..{MAX_STEP:g} for a move, "
                           f"up to {MAX_PLAN:g} for a hover step with all sticks 0).", 0.05, MAX_PLAN),
        "ramp_s": _num("Optional: seconds to blend smoothly from the previous sticks into these. The blend is "
                       "centred on the step boundary and never changes how far or how much this step moves "
                       "or turns: plan duration_s as if the sticks jumped. 0 = jump at once (default).",
                       0, MAX_STEP),
    }


def tool_specs(max_stick, vision):
    def fn(name, desc, props=None, required=()):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": list(required)}}}
    stick = _stick_props(max_stick)
    tools = [
        fn("takeoff", "Start the motors and climb to the drone's default hover height (~1 m). "
                      f"Returns after ~{TAKEOFF_WAIT:g} s once it has settled."),
        fn("land", "Descend and land at the current spot, then stop the motors."),
        fn("emergency_stop", "Cut the motors immediately. The drone FALLS. Only for emergencies or "
                             "when explicitly asked."),
        fn("calibrate_gyro", "Level the gyro. Only on the ground, on a flat surface."),
        fn("move", "Hold the sticks at exact percentages for an exact time, then recenter (hover). "
                   "Axes can be combined, e.g. forward and turn together to curve.", stick, ["duration_s"]),
        fn("sequence", "Fly a whole plan in one call: steps run back to back with exact timing and no pause "
                       f"between them (up to {MAX_PLAN:g} s total), then the sticks recenter. A step is "
                       "either sticks + duration_s, or {\"action\": \"takeoff\"} / {\"action\": \"land\"}, "
                       "so a command can start on the ground and finish landed. The drone hovers in place "
                       "while you are thinking between calls and drifts, so one sequence is far more precise "
                       "than several calls.",
           {"steps": {"type": "array", "minItems": 1, "maxItems": 100,
                      "items": {"type": "object", "properties": {
                          "action": {"type": "string", "enum": ["takeoff", "land"],
                                     "description": f"Take off (~{TAKEOFF_WAIT:g} s, then hovering at ~1 m) or "
                                                    f"land (~{LAND_WAIT:g} s). Leave out for a stick step."},
                          **stick}}}},
           ["steps"]),
        fn("hover", "Hold position (centered sticks, altitude hold) for a time.",
           {"duration_s": _num(f"Seconds, 0.05..{MAX_PLAN:g}.", 0.05, MAX_PLAN)}, ["duration_s"]),
        fn("fly_to", "Fly toward a tracked object (by id from get_status) or a picture point, keeping the "
                     "same height. Forward camera: centers it then approaches until it looks twice as big, "
                     "or - for an object with depth, if stop_at_m is given - until its CAUTIOUS distance is "
                     "down to stop_at_m (checked 20x a second). Returns arrived / lost / timeout.",
           {"object_id": {"type": "integer", "description": "Object id from get_status."},
            "stop_at_m": _num("With object_id: stop when the object's cautious camera-estimated distance "
                              "reaches this (m, 0.4-3). Use >= 0.8 unless asked to get very close.", 0.4, 3),
            "x": _num("Picture point, 0 = left edge, 1 = right edge (instead of object_id).", 0, 1),
            "y": _num("Picture point, 0 = top edge, 1 = bottom edge (instead of object_id).", 0, 1)}),
        fn("orbit", "Circle a person, camera on them the whole time (\"do a 360 around me\"). Finds them first "
                    "(turns on the spot if nobody's in view), then strafes sideways while turning to keep them "
                    "centred and keeping the same distance. Returns done / lost / not found / timeout.",
           {"object_id": {"type": "integer", "description": "A human's id from get_status. Leave out: the biggest "
                                                          "person in view, or search for one."},
            "direction": {"type": "string", "enum": ["right", "left"],
                          "description": "Which way to move round them (default right)."},
            "degrees": _num("How far round, 30..720 (default 360).", 30, 720)}),
        fn("set_trim", "Set the permanent stick offsets that cancel drift, and/or turn auto-trim on or off. "
                       "Auto-trim normally does this by itself from the video.",
           {"right": _num("Roll trim %, -25..25.", -25, 25), "forward": _num("Pitch trim %, -25..25.", -25, 25),
            "up": _num("Throttle trim %, -10..10.", -10, 10),
            "auto": {"type": "boolean", "description": "Auto-trim on (true) or off (false)."}}),
        fn("get_status", "Current state: airborne, link, sticks, trim, video drift estimate, tracked "
                         "objects with positions and time to contact, obstacle warning."),
    ]
    if vision:
        tools.append(fn("look", "See the current frame from the drone's forward camera, with tracked objects "
                                "outlined and labelled #id (use the id with fly_to). Use it to find things the "
                                "object list doesn't name, or to plan a way around an obstacle."))
    return tools


def system_prompt(max_stick, vision):
    notes = ""
    try:
        with open(NOTES_FILE) as f:
            notes = "\n\nPilot's calibration notes for this drone (trust these over the defaults above):\n" + f.read()
    except OSError:
        pass
    return f"""You are the autopilot of a small indoor Wi-Fi toy quadcopter. The human types commands; you fly by calling tools. Each tool blocks until its manoeuvre is finished and then reports the result.

How the drone works:
- It has no GPS and sends no telemetry. All you know comes from get_status (estimated from the camera video){" and look" if vision else ""}.
- Stick values are % of full deflection, -100..100, capped at ±{max_stick:g}% here. 0 on every axis = hover. Throttle uses altitude hold: up=0 keeps the height, + climbs, - descends.
- Axes: right (+right / -left slide), forward (+forward / -back), up (+climb / -descend), turn (+clockwise / -counter-clockwise yaw). Directions are relative to where the drone's nose points.
- Speed is not measured. Rough defaults for a toy drone indoors: forward/right 30% ≈ slow (~0.5 m/s), 50% ≈ brisk (~1 m/s); up 50% ≈ 0.5 m/s; turn 50% ≈ 90° per second. Below ~15% the drone may not respond at all. It keeps drifting briefly after the sticks recenter, so leave 0.5-1 s of hover between moves that must not blend.
- For distances and angles, convert to % and seconds with those numbers (or the pilot's notes) and say what you assumed.

Rules:
- Take off before any movement. Land when the human asks or when you are done with a task that started on the ground — but if they only asked for a manoeuvre while already flying, stay hovering.
- Fly each command as ONE `sequence` call whenever you can, including the takeoff and land steps: the drone drifts while it hovers waiting for you, so every extra call costs accuracy. Don't call get_status first unless you need it.
- A step's duration_s is the full-stick time: distance ≈ speed × duration_s and angle ≈ turn rate × duration_s. Ramps (ramp_s 0.2-0.4) make moves smoother without changing those amounts.
- Put a 0.5-1 s hover step between moves that must not blend (e.g. the sides of a square).
- If the human asks to fly forward a long way or toward something, call get_status first: if obstacle_ahead is true or something in_path has seconds_to_contact under 2, don't fly into it; tell the human.
- If a tool reports "aborted" the human took over: stop calling movement tools and explain what happened.
- emergency_stop drops the drone: only for a real emergency or when explicitly asked.

Distances (depth) - treat with caution:
- Objects may have distance_m, distance_m_cautious (20% nearer), bearing_deg, position_m, and get_status has depth.ahead_m (nearest thing straight ahead). They are ESTIMATED by a neural network from ONE camera and scaled by a hand calibration. They can be off by 20-50%, worse when the calibration is wrong, and they jump around on plain walls, glass, mirrors and in dim light. depth.trust and depth.why say how much to believe them right now; trust is never "high".
- Plan with distance_m_cautious, and stop at least 0.5 m short of it. For anything more than ~1.5 m away, go part of the way, check get_status again, then continue.
- If depth disagrees with other cues (seconds_to_contact, how big the object looks, the camera frame), believe whichever says the thing is CLOSER.
- When depth.trust is "low", or a tool result says DEPTH DISAGREES, go half as far per step and re-check, and tell the human the distances look unreliable.
- To approach an object and stop short of it, use fly_to with object_id and stop_at_m: that checks the distance 20x a second.
- "Do a 360 around me", "circle me", "orbit me": take off if needed, then call orbit (it finds the person itself). Don't fly a circle with stick steps.

Automatic safety (runs 20x a second, faster than you can):
- Obstacle guard: while a step flies forward, anything in the flight path that is very close or under 1.5 s away stops the plan at once, and so do the camera distance straight ahead staying under ~1 m and losing the video. The tool result then says "aborted: obstacle: ..." with stopped_during (which step), the objects and{" a camera frame" if vision else " the status"}. The drone is hovering where it stopped. Replan from there: go around (e.g. slide right 1 s, forward, slide back left) or stop and tell the human — never just repeat the same forward move.
- Position hold: during hover steps and while you think, the drone pushes against any slide it sees in the video. Hover steps between moves therefore also settle the drone, so keep them.
- Self-correction: the hold raises its own push limit against strong drift and lands the drone if it still can't stop it; it lands after 5 s without video and when the flight time limit (battery) is reached. These show up in get_status as safety_alerts / auto_landing. After an automatic landing, explain why and don't take off again unless the human asks.
- Safe area: every plan is checked against a dead-reckoned position (estimated_position_m in get_status); plans that would go further from the take-off spot than the safe area, above the ceiling or too near the floor are refused with the reason. Make them smaller rather than retrying the same plan.
- A guard stop is not the human taking over: after one you may keep flying. After "movement stopped" you may not.
- Keep replies to the human short: what you did, what you assumed.{notes}"""


SPEED_PER_PCT = 0.025   # m/s per % stick past STICK_DEAD: the rough speeds the LLM is told
STICK_DEAD = 10.0
CLIMB_PER_PCT = 0.01    # m/s per % throttle
TURN_PER_PCT = 1.8      # deg/s per % yaw
HOVER_HEIGHT = 1.0      # m after take off


class Envelope:
    """Dead-reckoned position from the plans flown, to refuse plans that would leave the room.

    Uses the same rough speeds the LLM is told, so it's an estimate that drifts (hover drift and
    keyboard flying aren't in it). It exists to catch plans that are plainly too big for a room:
    more than max_distance from where it first took off, above the ceiling, or near the floor."""

    def __init__(self, max_distance=3.0, ceiling=2.0, floor=0.3):
        self.max_distance, self.ceiling, self.floor = max_distance, ceiling, floor
        self.x = self.y = self.z = self.heading = 0.0

    def pose(self):
        return {"right": round(self.x, 2), "forward": round(self.y, 2), "height": round(self.z, 2),
                "heading_deg": round(self.heading % 360)}

    def limits(self):
        return {"max_distance_from_takeoff_m": self.max_distance or None, "ceiling_m": self.ceiling or None,
                "lowest_m": self.floor}

    def landed(self):
        self.z = 0.0

    def _run(self, steps, until=None, check=True):
        """Advance a copy of the pose through the steps; returns (pose, problem or None)."""
        x, y, z, hd = self.x, self.y, self.z, self.heading
        t = 0.0
        for i, st in enumerate(steps):
            if st.get("action") == "takeoff":
                z = HOVER_HEIGHT
            elif st.get("action") == "land":
                z = 0.0
            dur = st["duration"] if until is None else max(0.0, min(st["duration"], until - t))
            lv = st["levels"]
            n = max(1, int(dur / 0.05))
            h = dur / n
            for _ in range(n):
                fwd, right = _speed(lv["y"]), _speed(lv["x"])
                r = math.radians(hd)
                x += (right * math.cos(r) + fwd * math.sin(r)) * h
                y += (-right * math.sin(r) + fwd * math.cos(r)) * h
                z += lv["z"] * CLIMB_PER_PCT * h
                hd += lv.get("yaw", 0.0) * TURN_PER_PCT * h
            t += dur
            if check:
                d = math.hypot(x, y)
                if self.max_distance and d > self.max_distance:
                    return (x, y, z, hd), (f"step {i + 1} would take the drone about {d:.1f} m from where it "
                                           f"took off; the safe area is {self.max_distance:g} m. Make it smaller.")
                if self.ceiling and z > self.ceiling:
                    return (x, y, z, hd), (f"step {i + 1} would climb to about {z:.1f} m; the ceiling limit is "
                                           f"{self.ceiling:g} m.")
                if z < self.floor and st.get("action") != "land" and z > 0 and i + 1 < len(steps) \
                        and steps[i + 1].get("action") != "land":
                    return (x, y, z, hd), (f"step {i + 1} would come down to about {z:.1f} m; below "
                                           f"{self.floor:g} m use a land step instead.")
            if until is not None and t >= until:
                break
        return (x, y, z, hd), None

    def check(self, steps):
        return self._run(steps)[1]

    def commit(self, steps, flown_s=None):
        (self.x, self.y, self.z, self.heading), _ = self._run(steps, flown_s, check=False)


def _speed(p):
    return math.copysign(max(0.0, abs(p) - STICK_DEAD) * SPEED_PER_PCT, p)


class Agent:
    """Runs one typed command at a time: LLM -> tool calls -> Bridge, until the LLM stops calling tools."""

    def __init__(self, llm, bridge, frame=lambda: None, vision=False, log=print, envelope=None, auto_gyro=True):
        self.llm, self.bridge, self.frame, self.vision, self.log = llm, bridge, frame, vision, log
        self.envelope = envelope or Envelope()
        self.gyro_due = auto_gyro       # calibrate the gyro before the first take off of the session
        self.tools = tool_specs(bridge.max_stick, vision)
        self.messages = [{"role": "system", "content": system_prompt(bridge.max_stick, vision)}]
        self.inbox = deque(maxlen=1)
        self.wake = threading.Event()
        self.cancelled = False
        self.busy = False
        self.phase, self.phase_since = "idle", time.time()   # idle | thinking | running (HUD)
        self.current_tool = None
        self.last_command = self.last_reply = None
        threading.Thread(target=self._worker, daemon=True).start()

    def _set_phase(self, phase, tool=None):
        self.phase, self.phase_since, self.current_tool = phase, time.time(), tool

    def submit(self, text):
        """Queue a command. A new one cancels whatever the AI is doing."""
        if self.busy:
            self.cancelled = True
            self.bridge.abort("new command", by_pilot=False)
        self.inbox.append(text)
        self.wake.set()

    def cancel(self, why="stop"):
        self.cancelled = True
        return self.bridge.abort(f'"{why}" typed')

    def _worker(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            while self.inbox:
                text = self.inbox.popleft()
                self.cancelled, self.busy = False, True
                self.overrides = self.bridge.overrides
                try:
                    self.handle(text)
                except Exception as e:                      # keep the prompt alive whatever happens
                    self.log(f"AI error: {e}")
                    self.last_reply = f"ERROR: {e}"
                    flightlog.error("agent", e)
                    self.bridge.note("AI error - see terminal")
                finally:
                    self.busy = False
                    self._set_phase("idle")

    def handle(self, text):
        self._trim_history()
        self.messages.append({"role": "user", "content": text})
        self.bridge.note(f"AI: {text[:40]}")
        self.last_command, self.last_reply = text, None
        flightlog.event("command", text=text)
        for _ in range(MAX_ROUNDS):
            self._set_phase("thinking")
            msg = self.llm.chat(self.messages, self.tools)
            if self.cancelled:
                flightlog.event("cancelled", where="after llm reply")
                return
            calls = msg.get("tool_calls") or []
            self.messages.append({"role": "assistant", "content": msg.get("content") or None,
                                  **({"tool_calls": calls} if calls else {})})
            if msg.get("content"):
                self.log(f"AI: {msg['content'].strip()}")
                self.last_reply = msg["content"].strip()
            if not calls:
                return
            images = []
            for call in calls:
                name = call.get("function", {}).get("name", "")
                raw = call.get("function", {}).get("arguments") or "{}"
                t0 = time.time()
                args = None
                if self.cancelled:
                    result = {"error": "skipped: cancelled by the pilot"}
                else:
                    self._set_phase("running", name)
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else raw
                        result = self.execute(name, args or {}, images)
                    except (ValueError, TypeError, KeyError) as e:
                        result = {"error": str(e)}
                flightlog.event("tool", name=name, args=args if args is not None else raw, result=result,
                                seconds=round(time.time() - t0, 2))
                self.log(f"  {name}({_short(raw)}) -> {_short(json.dumps(result))}")
                self.messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                      "content": json.dumps(result)})
            if images:
                self.messages.append({"role": "user", "content": [
                    {"type": "text", "text": "Camera frame (tracked objects outlined with their ids):"}, *images]})
            if self.cancelled:
                return
        self.log("AI: too many steps for one command, stopped")

    def _trim_history(self):
        for m in self.messages[1:]:             # old camera frames cost a lot of tokens
            if isinstance(m.get("content"), list):
                m["content"] = "(camera frame from look(), no longer attached)"
        if len(self.messages) > HISTORY:
            keep = self.messages[-HISTORY:]
            while keep and keep[0]["role"] != "user":   # never start on an orphaned tool reply
                keep.pop(0)
            self.messages = self.messages[:1] + keep

    # ---- tools
    def execute(self, name, a, images):
        st = self.bridge.snapshot()
        flying = st.get("airborne", False)
        if not flying:
            self.envelope.landed()
        if name == "get_status":
            if not st:
                return {"error": "no status yet (video loop not started)"}
            return {**st, "estimated_position_m": self.envelope.pose(), "safe_area": self.envelope.limits()}
        if name in ("move", "hover", "fly_to", "sequence", "takeoff") and st.get("headless_mode"):
            return {"error": "headless mode is on (H key): then 'forward' means away from the pilot, not where "
                             "the nose points, and your plans would go the wrong way. Ask the human to press H "
                             "to turn it off."}
        if name == "takeoff" and self.bridge.overrides == self.overrides:
            if flying:
                return {"result": "already flying"}
            self._preflight()
            self.bridge.request("action", "takeoff")
            self.envelope.z = HOVER_HEIGHT
            return self._settle(TAKEOFF_WAIT, "took off")
        if name == "land":
            self.bridge.abort("landing", by_pilot=False)
            self.bridge.request("action", "land")
            self.envelope.landed()
            return self._settle(LAND_WAIT, "landed")
        if name == "emergency_stop":
            self.bridge.abort("motors off", by_pilot=False)
            self.bridge.request("action", "stop")
            return {"result": "motors off"}
        if name == "calibrate_gyro":
            if flying:
                return {"error": "only on the ground"}
            self.bridge.request("action", "gyro")
            return self._settle(GYRO_WAIT, "gyro calibrated")
        if name == "set_trim":
            trims = {AXES[k]: float(a[k]) for k in ("right", "forward", "up") if a.get(k) is not None}
            self.bridge.request("trim", trims, a.get("auto"))
            time.sleep(0.1)
            s = self.bridge.snapshot()
            return {"trim_pct": s.get("trim_pct"), "auto_trim": s.get("auto_trim")}
        if name == "look":
            jpg = self.frame() if self.vision else None
            if not jpg:
                return {"error": "no video frame" if self.vision else "vision is off (--ai-vision)"}
            self._attach(jpg, st, images)
            return {"result": "frame attached in the next message", "objects": st.get("objects", [])}
        if name in ("move", "sequence", "hover", "fly_to", "orbit", "takeoff") and self.bridge.overrides != self.overrides:
            return {"error": f"movement stopped ({self.bridge.lock_reason}): no more movement until the "
                             "human types a new command"}
        if name in ("move", "hover", "fly_to", "orbit") and not flying:
            return {"error": "not flying: call takeoff first"}
        if name == "fly_to":
            return self._fly_to(a, st)
        if name == "orbit":
            return self._orbit(a)
        clamped = set()
        if name == "move":
            steps = [self._step(a, clamped)]
        elif name == "hover":
            d = float(a.get("duration_s", 0))
            if not 0.05 <= d <= MAX_PLAN:
                raise ValueError(f"duration_s must be 0.05..{MAX_PLAN:g}")
            steps = [self._hold(d)]
        elif name == "sequence":
            steps = []
            air = flying
            for i, d in enumerate(a.get("steps") or []):
                action = d.get("action")
                if action == "takeoff":
                    if air:
                        raise ValueError(f"step {i + 1}: already flying, remove the takeoff")
                    steps.append(self._hold(TAKEOFF_WAIT, "takeoff"))
                    air = True
                elif action == "land":
                    if not air:
                        raise ValueError(f"step {i + 1}: land while not flying")
                    steps.append(self._hold(LAND_WAIT, "land"))
                    air = False
                elif action:
                    raise ValueError(f"step {i + 1}: unknown action {action!r}")
                elif not air:
                    raise ValueError(f"step {i + 1}: moves while on the ground - start with a takeoff step")
                else:
                    steps.append(self._step(d, clamped))
            if not steps:
                raise ValueError("steps is empty")
            total = sum(s["duration"] for s in steps)
            if total > MAX_PLAN:
                raise ValueError(f"sequence is {total:.1f} s, the limit is {MAX_PLAN:g} s: split it")
        else:
            return {"error": f"unknown tool {name}"}
        problem = self.envelope.check(steps)
        if problem:
            return {"error": "plan refused: " + problem, "estimated_position_m": self.envelope.pose(),
                    "safe_area": self.envelope.limits()}
        if steps[0].get("action") == "takeoff":
            self._preflight()
        t0 = time.time()
        before = (st.get("depth") or {}).get("ahead_m")
        pose0 = (self.envelope.x, self.envelope.y, self.envelope.heading)
        result = self.bridge.run(steps)
        self.envelope.commit(steps, None if result == "done" else (self.bridge.ended or {}).get("elapsed"))
        out = {"result": result, "flown_s": round(time.time() - t0, 2),
               "estimated_position_m": self.envelope.pose()}
        if clamped:
            out["clamped_to_max"] = sorted(clamped)
        after = self.bridge.snapshot()
        out.update({k: after.get(k) for k in ("airborne", "drift", "obstacle_ahead")})
        check = depth_check(before, (after.get("depth") or {}).get("ahead_m"), pose0,
                            (self.envelope.x, self.envelope.y, self.envelope.heading))
        if check:
            out["depth_check"] = check
        if result != "done":
            at = self.bridge.ended or {}
            out["stopped_during"] = f"step {at.get('step')} of {at.get('steps')} ({at.get('what')}), " \
                                    f"{at.get('elapsed')} s into {at.get('total')} s"
            out["objects"] = after.get("objects", [])
            if self.vision and ("obstacle" in result or "see ahead" in result):
                jpg = self.frame()
                if jpg:
                    self._attach(jpg, after, images)
                    out["camera"] = "frame from when it stopped attached in the next message"
        return out

    def _attach(self, jpg, status, images):
        images.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(annotate(jpg, status.get("objects", []))).decode()}})

    @staticmethod
    def _hold(seconds, action=None):
        return {"levels": dict.fromkeys(AXES.values(), 0.0), "duration": seconds, "ramp": 0.0, "action": action}

    def _step(self, d, clamped):
        dur = float(d.get("duration_s", 0))
        still = not any(float(d.get(n) or 0) for n in AXES)
        longest = MAX_PLAN if still else MAX_STEP           # a hover step may be long, a move may not
        if not 0.05 <= dur <= longest:
            raise ValueError(f"duration_s must be 0.05..{longest:g}" + ("" if still else " for a move: split it"))
        levels = {}
        for name, key in AXES.items():
            v = float(d.get(name) or 0)
            if abs(v) > self.bridge.max_stick:
                clamped.add(name)
                v = math.copysign(self.bridge.max_stick, v)
            levels[key] = v
        return {"levels": levels, "duration": dur, "ramp": min(dur, max(0.0, float(d.get("ramp_s") or 0)))}

    def _preflight(self):
        """Once per session, on the ground: level the gyro, so the drone doesn't turn by itself (a turn
        looks like a sideways slide to the camera, which the position hold would then 'correct')."""
        if not self.gyro_due:
            return
        self.gyro_due = False
        self.log("AI: calibrating the gyro before the first take off (keep the drone flat and still)")
        flightlog.event("auto_gyro_calibrate")
        self.bridge.note("calibrating gyro before take off")
        self.bridge.request("action", "gyro")
        self._settle(GYRO_WAIT, "gyro calibrated")

    def _settle(self, seconds, done):
        end = time.time() + seconds
        while time.time() < end:
            if self.cancelled:
                return {"result": "aborted: cancelled by the pilot"}
            time.sleep(0.05)
        s = self.bridge.snapshot()
        return {"result": done, "airborne": s.get("airborne")}

    def _orbit(self, a):
        oid = int(a["object_id"]) if a.get("object_id") is not None else None
        self.bridge.request("orbit", oid, a.get("direction", "right"), float(a.get("degrees", 360)))
        if not self.bridge.wait_until(lambda s: s.get("orbit") in ("finding", "orbiting"), 1.5):
            return {"error": "couldn't start the orbit (no video?)"}
        self.bridge.wait_until(lambda s: s.get("orbit") not in ("finding", "orbiting") or self.cancelled, 90)
        s = self.bridge.snapshot()
        return {"result": s.get("orbit"), "turned_deg": s.get("orbit_turned_deg")}

    def _fly_to(self, a, st):
        if a.get("object_id") is not None:
            obj = next((o for o in st.get("objects", []) if o["id"] == int(a["object_id"])), None)
            if obj is None:
                return {"error": f"no object #{a['object_id']} in view", "objects": st.get("objects", [])}
            u, v = obj["x"], obj["y"]
        elif a.get("x") is not None and a.get("y") is not None:
            u, v = float(a["x"]), float(a["y"])
        else:
            raise ValueError("give object_id, or both x and y")
        stop_at = None
        if a.get("stop_at_m") is not None and a.get("object_id") is not None:
            stop_at = min(3.0, max(0.4, float(a["stop_at_m"])))
        oid = int(a["object_id"]) if a.get("object_id") is not None else None
        self.bridge.request("goto", min(1.0, max(0.0, u)), min(1.0, max(0.0, v)), oid, stop_at)
        if not self.bridge.wait_until(lambda s: s.get("fly_to") == "tracking", 1.5):
            return {"error": "couldn't lock on (no video?)"}
        self.bridge.wait_until(lambda s: s.get("fly_to") != "tracking" or self.cancelled, GOTO_TIMEOUT)
        s = self.bridge.snapshot()
        out = {"result": s.get("fly_to"), "obstacle_ahead": s.get("obstacle_ahead")}
        if oid is not None:
            obj = next((o for o in s.get("objects", []) if o["id"] == oid), None)
            if obj and obj.get("distance_m") is not None:
                out["object_distance_now"] = {k: obj[k] for k in ("distance_m", "distance_m_cautious", "distance_is")}
        return out


def depth_check(before, after, pose0, pose1):
    """After a plan: did the camera's distance-ahead shrink by about how far we (think we) flew forward?
    Both numbers are estimates, so only a clear disagreement is reported."""
    if before is None or after is None:
        return None
    if abs(((pose1[2] - pose0[2]) + 180) % 360 - 180) > 15:
        return None                                      # turned: it's looking at something else
    h = math.radians(pose0[2])
    fwd = (pose1[0] - pose0[0]) * math.sin(h) + (pose1[1] - pose0[1]) * math.cos(h)
    if fwd < 0.3:
        return None
    closed = before - after
    agrees = abs(closed - fwd) <= max(0.3, 0.5 * fwd)
    flightlog.event("depth_check", flown_forward_m=round(fwd, 2), ahead_before=before, ahead_after=after,
                    agrees=agrees)
    if agrees:
        return f"depth agreed: flew ~{fwd:.1f} m forward and the distance ahead shrank by {closed:.1f} m"
    return (f"DEPTH DISAGREES: flew ~{fwd:.1f} m forward (estimated from the sticks) but the distance ahead "
            f"changed from {before:.1f} to {after:.1f} m. Either the depth calibration or the speed guess is off: "
            f"trust the distances less and move in shorter steps.")


def annotate(jpg, objects):
    """Draw each tracked object's box and "#id label" on the frame, so the LLM can name them."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return jpg
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpg
    h, w = img.shape[:2]
    for o in objects:
        x0, y0 = int((o["x"] - o["width"] / 2) * w), int((o["y"] - o["height"] / 2) * h)
        x1, y1 = int((o["x"] + o["width"] / 2) * w), int((o["y"] + o["height"] / 2) * h)
        color = (60, 60, 240) if o.get("in_path") else (80, 220, 80)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        label = f"#{o['id']} {o['label']}" + (f" ~{o['distance_m']:.1f}m?" if o.get("distance_m") is not None else "")
        cv2.putText(img, label, (x0 + 3, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _short(s, n=160):
    s = s if isinstance(s, str) else json.dumps(s)
    return s if len(s) <= n else s[:n] + "..."


def console(agent, bridge, direct, prompt="ai> "):
    """Read commands from the terminal. stop / land / kill bypass the LLM."""
    print('Type a command for the AI pilot ("stop", "land", "kill" act at once, no LLM).', flush=True)
    try:
        import voice
        threading.Thread(target=voice.load, daemon=True).start()   # warm up while you get ready
        print("Or press Enter on an empty line and say it.", flush=True)
    except ImportError:
        voice = None
    while True:
        try:
            text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not text and voice:
            try:
                text = voice.listen(lambda m: print(m, flush=True))
            except Exception as e:      # no mic permission, no model download yet, ...
                print(f"  voice failed: {e}", flush=True)
                continue
            text = voice.direct_word(text, DIRECT) or text
        word = text.lower()
        if not text:
            continue
        if word in DIRECT:
            agent.cancel(word)
            if word != "stop":
                direct("stop" if word == "kill" else "land")
            print(f"  {word}: done", flush=True)
        else:
            agent.submit(text)


def start_console(agent, bridge, direct):
    threading.Thread(target=console, args=(agent, bridge, direct), daemon=True).start()


# ---------------- dry run: a pretend drone loop, prints the bytes that would be sent ----------------
class DryRun:
    def __init__(self, bridge, on_tick=None, quiet=False, sup=None, perceive=None, autotrim=False, depth=None):
        """on_tick(sticks, now) is called every 50 ms with the Sticks that would be sent.
        sup: a supervisor.Supervisor to run (guard + hold); perceive(now) -> (drift vel, valid, objects)
        stands in for the video (sim_flight.py simulates it)."""
        from drone import Pilot                 # lazy: drone.py imports this module
        from protocol import control_frame, FLAG_TAKEOFF, FLAG_LAND, FLAG_STOP, FLAG_GYRO_CAL
        self.control_frame = control_frame
        self.flags = {"takeoff": FLAG_TAKEOFF, "land": FLAG_LAND, "stop": FLAG_STOP, "gyro": FLAG_GYRO_CAL}
        self.bridge, self.pilot = bridge, Pilot()
        self.pilot.source = bridge.levels       # same as drone.py --ai
        self.flight_start = None
        self.t0 = time.time()
        self.on_tick, self.quiet = on_tick, quiet
        self.sup, self.perceive = sup, perceive
        self.depth = depth                      # supervisor.DepthTrust, when perceive() has depth
        if sup:
            sup.pilot = self.pilot                         # hold feeds steady pushes into the trim
        self.last_tel = 0.0
        self.last_input = 0.0
        self.autotrim = None
        if autotrim:                            # the real auto-trim, fed by perceive()'s drift
            import drift
            self.autotrim = drift.AutoTrim(self.pilot, True)
            if sup:
                sup.autotrim = self.autotrim

    def direct(self, word):
        self.bridge.request("action", word)

    def loop(self):
        last = None
        while True:
            now = time.time()
            for req in self.bridge.take_requests():
                if req[0] == "action":
                    self.pilot.pulse(self.flags[req[1]])
                    self.flight_start = (self.flight_start or now) if req[1] == "takeoff" else (
                        self.flight_start if req[1] == "gyro" else None)
                elif req[0] == "trim":
                    self.pilot.set_trims(req[1])
                elif req[0] == "goto":
                    print(f"  [dry run] fly_to {req[1]:.2f},{req[2]:.2f} - no video, reporting lost", flush=True)
            vel, valid, objs = (self.perceive(now) if self.perceive
                                else ({"x": 0.0, "y": 0.0, "z": 0.0}, False, []))
            age = self.perceive.age(now) if hasattr(self.perceive, "age") else 0.05
            ahead = self.perceive.ahead(now) if hasattr(self.perceive, "ahead") else None
            if self.sup:
                airborne_for = None if self.flight_start is None else now - self.flight_start
                self.pilot.set_hold(self.sup.update(now, airborne_for, False, False, age, vel, valid, objs, ahead))
                if self.depth:
                    self.depth.update(now, ahead[0] if ahead else None, self.sup.hold_state == "holding")
            lv = self.bridge.levels(now)
            if lv is None:
                self.pilot.update(dict.fromkeys("xyz", 0), 0.05)
            else:
                self.pilot.steer(lv)
                step = self.bridge.progress(now)
                if step is None or step["kind"] != "hover":
                    self.last_input = now
            if self.autotrim:
                from types import SimpleNamespace
                airborne_for = None if self.flight_start is None else now - self.flight_start
                self.autotrim.update(SimpleNamespace(vel=vel, valid=valid), 0.05, airborne_for, now - self.last_input)
            sticks = self.pilot.sticks()
            if self.on_tick:
                self.on_tick(sticks, now)
            frame = self.control_frame(sticks)
            if frame != last and not self.quiet:
                s = self.pilot.sticks()
                print(f"  [{now - self.t0:6.2f}s] {frame.hex(' ')}   roll {s.roll} pitch {s.pitch} "
                      f"thr {s.throttle} yaw {s.yaw} flags {s.flags:#04x}", flush=True)
                last = frame
            levels = self.pilot.levels()
            levels["yaw"] = self.pilot.yaw_level()
            objects = objects_json(objs)
            if now - self.last_tel >= 0.2:
                self.last_tel = now
                flightlog.event("tel", airborne=self.flight_start is not None, levels=levels,
                                sticks=[sticks.roll, sticks.pitch, sticks.throttle, sticks.yaw, sticks.flags],
                                drift=vel, drift_valid=valid, objects=objects,
                                trim=self.pilot.trims(), auto=self.autotrim.state if self.autotrim else "off",
                                **(self.sup.tel_fields(now) if self.sup else {}))
            self.bridge.publish({
                "airborne": self.flight_start is not None,
                "flight_s": None if self.flight_start is None else round(now - self.flight_start, 1),
                "link": "DRY RUN", "sticks_pct": {k: round(levels[v]) for k, v in AXES.items()},
                "trim_pct": {k: round(self.pilot.trims()[v], 1) for k, v in AXES.items() if v != "yaw"},
                "auto_trim": "off", "drift": "unknown (dry run)" if not valid else
                {k: round(v, 3) for k, v in vel.items()}, "fly_to": "lost",
                "objects": objects, "obstacle_ahead": any(o["in_path"] for o in objects),
                **(self.sup.status() if self.sup else {}),
                **(self.depth.status(ahead) if self.depth else {})})
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="*", help="with --dry-run: a command to run (else prompts)")
    ap.add_argument("--list-models", action="store_true", help="print the model ids your key can use")
    ap.add_argument("--dry-run", action="store_true", help="no drone: print the control frames instead")
    ap.add_argument("--model", help="override SPAR_MODEL")
    ap.add_argument("--max-stick", type=float, default=60.0, help="stick cap in %% (default 60)")
    args = ap.parse_args()
    llm = LLM(model=args.model)
    if args.list_models:
        for m in llm.models():
            print(m)
        return
    if not args.dry_run:
        sys.exit("To fly the real drone: python drone.py --ai   (or add --dry-run here to test without one)")
    import supervisor
    bridge = Bridge(args.max_stick)
    print("flight log:", os.path.relpath(flightlog.LOG.open("dryrun")), flush=True)
    sim = DryRun(bridge, sup=supervisor.Supervisor(bridge))
    threading.Thread(target=sim.loop, daemon=True).start()
    agent = Agent(llm, bridge)
    print(f"dry run with {llm.model} at {llm.base}", flush=True)
    if args.command:
        agent.submit(" ".join(args.command))
        time.sleep(0.2)
        while agent.busy or agent.inbox:
            time.sleep(0.1)
        time.sleep(0.3)
        return
    console(agent, bridge, sim.direct)


if __name__ == "__main__":
    main()
