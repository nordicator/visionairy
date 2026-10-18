"""Fast safety and stabilisation layer under the LLM pilot, run every video frame.

The LLM takes seconds to answer, so it can't react to anything in flight. This layer does, and
fixes what it can by itself:

  Obstacle guard  While an AI plan step flies forward, anything tracked in the flight path that is
                  very close or under GUARD_TTC s away stops the plan, and so does stale video or
                  the camera losing all texture (a blank wall filling the view, or the room too
                  dark). The LLM is told why and gets a camera frame, so it can replan.
  Position hold   While the drone should be still (a plan's hover steps, or waiting for the LLM),
                  it pushes against any slide the video measures. A steady drift builds up an
                  integral term (so it ends up cancelled, not just resisted, and nothing is saved to
                  trim.json), and something closing in ahead makes it back away. It also holds through
                  the take-off step and while landing. It pauses after turns (a turn looks like a
                  sideways slide) and whenever it can't trust the video.
  Self-checks     Per axis (x = sideways, y = forward/back):
                  - wobbling back and forth -> its gain is cut
                  - the slide growing although it pushes against it (the estimate's direction is
                    wrong on this drone) -> the trim it fed is taken back, auto-trim stops on that
                    axis, and the axis is flipped; if flipped is wrong too, it's switched off
                  - a slide it can't stop at full push -> its push limit is raised step by step,
                    and if that still can't stop it, the drone lands
  Watchdogs       Video gone for LOST_LAND_S while flying -> land. Flight time past FLIGHT_WARN_S ->
                  warning, past FLIGHT_MAX_S -> land (toy batteries die suddenly).

Every action is written to the flight log, shown on the HUD, and reported to the LLM in its status.
"""
import os
from collections import deque

import flightlog

GUARD_MIN_FORWARD = 8.0    # % forward stick from which the guard watches the path
GUARD_TTC = 1.5            # s, stop if something in the path is this close in time (obstacles.py flags < 2 s)
GUARD_MAX_AGE = 0.5        # s, stop a forward step if the video is older than this (can't see ahead)
GUARD_BLIND_S = 0.6        # s of no texture while flying forward before stopping (blank wall / dark)
DEPTH_GUARD_S = 0.3        # s the camera depth must keep saying "under ~1 m straight ahead" before stopping
DEPTH_UNSTEADY = 0.25      # relative spread of the distance ahead while hovering that counts as unreliable
DEPTH_OLD_DAYS = 14        # a depth calibration older than this is worth a mention

HOLD_GAIN = 120.0          # % stick per frame-width/s of measured slide, on top of HOLD_MIN
HOLD_MIN = 12.0            # % stick: toy drones ignore smaller inputs, so any push starts here
HOLD_MAX = 20.0            # % stick, normal cap on the hold correction
HOLD_DEADBAND = 0.01       # frame-widths/s treated as still (a bit above drift.DEADBAND: noise)
HOLD_TAU = 0.25            # s, smoothing on the correction
HOLD_AFTER_TAKEOFF = 1.0   # s after take off before holding (the first jump off the ground)
HOLD_AFTER_TURN = 1.2      # s after a turn before trusting the slide estimate again
HOLD_I_RATE = 0.5          # per second: this fraction of the push builds up in the integral term
HOLD_WHILE_LANDING = 2.0   # s to keep holding after a landing starts (it's still descending)

BACKOFF_TTC = 1.5          # s: something in the path closing in this fast while holding -> back away

OSC_WINDOW = 4.0           # s: this many reversals of the slide inside the window = wobbling
OSC_FLIPS = 4
OSC_CUT = 0.6              # ... and gain and minimum push are multiplied by this
GAIN_FLOOR = 0.25          # never below this fraction of the starting gain
MIN_FLOOR = 6.0            # % stick, lowest minimum push after cuts

WORSE_WINDOW = 2.0         # s of pushing one way before judging whether it helped
WORSE_FACTOR = 1.6         # slide this many times faster than when the push started = wrong direction
WORSE_MIN_V = 0.03         # frame-widths/s: ignore tiny slides when judging

RUNAWAY_V = 0.05           # frame-widths/s of slide that counts as strong drift
STILL_SLIDING = 0.02       # frame-widths/s: at full push and still sliding this fast = needs more push
RUNAWAY_STEP_S = 2.0       # s of sliding at full push before the cap is raised ...
HOLD_CAP_STEP = 5.0        # ... by this much ...
HOLD_CAP_LIMIT = 35.0      # ... up to this
RUNAWAY_LAND_S = 5.0       # s of strong drift at the highest cap before landing
UNFIXABLE_LAND_S = 8.0     # s of drift on a switched-off axis before landing
HEAVY_CORRECTION = 30.0    # % stick of trim + push on one axis that's worth a warning at once ...
SUSTAINED_CORRECTION = 20.0  # ... or this much held for SUSTAINED_S (a 0.15 m/s draft needs ~16). A drone
SUSTAINED_S = 6.0          # turning by itself looks exactly like this: it can't be told apart in the air

LOST_LAND_S = 5.0          # s without video while flying before landing
FLIGHT_WARN_S = 300.0      # s of flight before the battery warning
FLIGHT_MAX_S = 420.0       # s of flight before an automatic landing (0 = never)

TEL_PERIOD = 0.2           # s between telemetry lines in the flight log (5 Hz)


class Axis:
    """Hold state for one axis: gain/limits it tuned for itself, and its self-check timers."""

    def __init__(self, gain, cap, minimum):
        self.gain0 = gain
        self.gain, self.cap, self.min = gain, cap, minimum
        self.enabled = True
        self.flipped = False            # the camera estimate turned out reversed on this axis: use -v
        self.flips = deque()            # times the push changed direction
        self.sign = 0                   # direction of the current push (0 = too small to count)
        self.last_dir = 0               # last clear direction of the measured slide, to count reversals
        self.push_since = None          # when the push started in this direction
        self.v_start = 0.0              # slide when it started
        self.strong_since = None        # strong drift since
        self.raised_at = 0.0            # last cap raise
        self.i = 0.0                    # integral term, % stick (steady drift), reset each flight
        self.heavy_since = None         # large correction held since


class Supervisor:
    def __init__(self, bridge, camera="forward", guard=True, hold=True, hold_gain=HOLD_GAIN,
                 hold_max=HOLD_MAX, hold_min=HOLD_MIN, lost_land_s=LOST_LAND_S, flight_max_s=FLIGHT_MAX_S,
                 flight_warn_s=FLIGHT_WARN_S):
        self.bridge, self.camera = bridge, camera
        self.guard_on, self.hold_on = guard, hold
        self.hold_max = hold_max
        self.hold_min = min(hold_min, hold_max)
        self.axes = {a: Axis(hold_gain, hold_max, self.hold_min) for a in ("x", "y")}
        self.lost_land_s, self.flight_max_s, self.flight_warn_s = lost_land_s, flight_max_s, flight_warn_s
        self.pilot = None                       # drone.Pilot: trims() / adjust_trim() / set_trims()
        self.autotrim = None                    # drift.AutoTrim: its .skip set
        self.takeoff_trim = {}                  # trims at take off, to restore an axis found reversed
        self.hold = {"x": 0.0, "y": 0.0}       # current correction, % stick
        self.guard_state, self.guard_detail = "standby", ""
        self.hold_state = "off" if not hold else "waiting"
        self.last_stop = None                  # (time, reason) of the last guard stop
        self.last_turn = 0.0
        self.blind_since = None
        self.depth_close_since = None
        self.ahead = None
        self.last_t = None
        self.was_airborne = False
        self.descend_until = 0.0               # keep holding while the drone comes down
        self.landing = None                    # reason, once an automatic landing was asked for
        self.alerts = {}                       # key -> (text, level "warn" | "act", since)

    # ---------------- alerts ----------------
    def _alert(self, key, text, level="warn", now=None):
        old = self.alerts.get(key)
        if old is None or old[0] != text:
            flightlog.event("alert", key=key, text=text, level=level)
            if level == "act" and (old is None or old[1] != "act"):
                self.bridge.note(text)
        self.alerts[key] = (text, level, old[2] if old else now)

    def _clear(self, key):
        if self.alerts.pop(key, None) is not None:
            flightlog.event("alert_clear", key=key)

    def _land(self, why, now):
        """Automatic landing: stop the plan (locking the AI out of moving) and land."""
        if self.landing:
            return
        self.landing = why
        self.bridge.abort(f"landing automatically: {why}", by_pilot=True)
        self.bridge.request("action", "land")
        self._alert("landing", f"AUTO-LANDING: {why}", "act", now)
        flightlog.event("auto_land", why=why)

    # ---------------- every frame ----------------
    def update(self, now, airborne_for, keys_held, goto_active, age, est_vel, est_valid, objects, ahead=None):
        """Run the guard, watchdogs and hold; return the hold correction {x, y} in % stick.
        ahead: (metres to the nearest thing straight ahead, whether that's close) from depth, or None."""
        self.ahead = ahead
        dt = 0.0 if self.last_t is None else max(0.0, min(0.2, now - self.last_t))
        self.last_t = now
        self._flight_changes(now, airborne_for)
        self._watchdogs(now, airborne_for, age)
        prog = self.bridge.progress(now)
        self._guard(now, prog, age, objects, est_vel, est_valid, ahead)
        prog = self.bridge.progress(now)                # the guard may just have stopped it
        self._hold(now, dt, prog, airborne_for, keys_held, goto_active, age, est_vel, est_valid, objects)
        return dict(self.hold)

    def _flight_changes(self, now, airborne_for):
        airborne = airborne_for is not None
        if airborne and not self.was_airborne:
            self.landing = None
            self._clear("landing")
            for ax in self.axes.values():            # fresh limits each flight; tuned gains and
                ax.cap, ax.i = self.hold_max, 0.0               # flipped / switched-off axes stay as learned
                ax.strong_since, ax.push_since, ax.sign = None, None, 0
            for key in ("battery", "video", "drift_x", "drift_y", "heavy_x", "heavy_y"):
                self._clear(key)
            if self.pilot:
                self.takeoff_trim = self.pilot.trims()
        elif self.was_airborne and not airborne:
            self.descend_until = now + HOLD_WHILE_LANDING
        self.was_airborne = airborne

    def _watchdogs(self, now, airborne_for, age):
        if airborne_for is None:
            self._clear("video")
            return
        lost = None if age is not None and age < 1.0 else (age if age is not None else airborne_for)
        if lost is None:
            self._clear("video")
        elif self.lost_land_s and lost >= self.lost_land_s:
            self._land(f"no video for {lost:.0f} s", now)
        else:
            self._alert("video", f"VIDEO LOST {lost:.1f}s" + (f" - landing at {self.lost_land_s:.0f}s"
                                                               if self.lost_land_s else ""), "warn", now)
        if self.flight_max_s and airborne_for >= self.flight_max_s:
            self._land(f"flight time limit {self.flight_max_s / 60:.1f} min (battery)", now)
        elif self.flight_warn_s and airborne_for >= self.flight_warn_s:
            m, s = divmod(int(airborne_for), 60)
            self._alert("battery", f"FLYING {m}:{s:02d} - battery may be low, land soon", "warn", now)

    def _guard(self, now, prog, age, objects, est_vel, est_valid, ahead=None):
        if not self.guard_on:
            state, detail = "off", ""
        elif self.camera != "forward":
            state, detail = "off", "down camera"
        elif prog is None:
            state, detail = "standby", ""
        elif prog["levels"]["y"] < GUARD_MIN_FORWARD:
            state, detail = "standby", "not flying forward"
        else:
            state, detail = "watching", "" if objects is not None else "object tracking is off: blind!"
            why, obj = None, None
            self.blind_since = None if est_valid else (self.blind_since or now)
            close = bool(ahead and ahead[1] and ahead[0] is not None)
            self.depth_close_since = (self.depth_close_since or now) if close else None
            if age is None or age > GUARD_MAX_AGE:
                why = f"can't see ahead (video {'missing' if age is None else f'{age:.1f} s old'})"
            elif self.blind_since and now - self.blind_since >= GUARD_BLIND_S:
                why = "can't see ahead: no texture in the camera view (blank wall very close, or too dark)"
            elif ahead and ahead[1] and ahead[0] is not None and self.depth_close_since \
                    and now - self.depth_close_since >= DEPTH_GUARD_S:
                why = f"camera depth: something about {ahead[0]:.1f} m straight ahead (an estimate)"
            else:
                threats = [o for o in objects or [] if o["threat"] and not o["lost"]]
                close = [o for o in objects or [] if not o["lost"] and o.get("ttc") is not None
                         and o["ttc"] < GUARD_TTC and _in_path(o)]
                found = threats or close
                if found:
                    obj = min(found, key=lambda o: o["ttc"] if o.get("ttc") is not None else 0.0)
                    when = f"{obj['ttc']:.1f} s to contact" if obj.get("ttc") is not None else "very close"
                    why = f"obstacle: {obj['label']} #{obj['id']} in the flight path, {when}"
            if why:
                self.bridge.abort(why, by_pilot=False)
                self.last_stop = (now, why)
                state, detail = "stopped", why
                flightlog.event("guard_stop", why=why, object=obj, plan=prog, drift=est_vel, age=age)
        if state != "watching":
            self.blind_since = self.depth_close_since = None
        if self.last_stop and now - self.last_stop[0] < 4.0 and state != "watching":
            state, detail = "stopped", self.last_stop[1]           # keep showing it for a while
        self.guard_state, self.guard_detail = state, detail

    def _hold(self, now, dt, prog, airborne_for, keys_held, goto_active, age, est_vel, est_valid, objects):
        if prog is not None and abs(prog["levels"].get("yaw", 0.0)) >= 0.5:
            self.last_turn = now
        still_step = prog is not None and prog["kind"] in ("hover", "takeoff", "land")
        idle = prog is None and not goto_active
        descending = airborne_for is None and now < self.descend_until
        reason = None
        if not self.hold_on:
            state = "off"
        elif airborne_for is None and not descending:
            state, reason = "waiting", "on the ground"
        elif airborne_for is not None and airborne_for < HOLD_AFTER_TAKEOFF:
            state, reason = "waiting", "leaving the ground"
        elif keys_held:
            state, reason = "waiting", "pilot flying"
        elif goto_active:
            state, reason = "waiting", "fly-to steering"
        elif not (still_step or idle):
            state, reason = "waiting", "plan moving"
        elif now - self.last_turn < HOLD_AFTER_TURN:
            state, reason = "waiting", "settling after turn"
        elif age is None or age > 0.5:
            state, reason = "waiting", "no video"
        elif not est_valid:
            state, reason = "waiting", "low texture"
        else:
            state = "holding"
        backing = None
        if state == "holding":
            k = 1.0 if dt <= 0 else min(1.0, dt / HOLD_TAU)
            closing = [o for o in objects or [] if not o["lost"] and _in_path(o)
                       and (o["threat"] or (o.get("ttc") is not None and o["ttc"] < BACKOFF_TTC))]
            if closing and self.camera == "forward":
                backing = min(closing, key=lambda o: o["ttc"] if o.get("ttc") is not None else 0.0)
            for a, ax in self.axes.items():
                v = -est_vel[a] if ax.flipped else est_vel[a]
                p = 0.0
                if ax.enabled and abs(v) >= HOLD_DEADBAND:
                    p = ax.min + ax.gain * (abs(v) - HOLD_DEADBAND)
                    p = -p if v > 0 else p
                if ax.enabled and not (a == "y" and backing is not None):
                    ax.i = max(-ax.cap, min(ax.cap, ax.i + HOLD_I_RATE * max(-ax.cap, min(ax.cap, p)) * dt))
                want = max(-ax.cap, min(ax.cap, p + ax.i)) if ax.enabled else 0.0
                if a == "y" and backing is not None:
                    want = -max(ax.cap, self.hold_min)          # back away from it
                self.hold[a] += (want - self.hold[a]) * k
                if ax.enabled:
                    self._self_check(a, ax, now, v)
                else:
                    self._unfixable(a, ax, now, v)
                if self.pilot:
                    total = self.pilot.trims()[a] + self.hold[a]
                    ax.heavy_since = (ax.heavy_since or now) if abs(total) >= SUSTAINED_CORRECTION else None
                    if abs(total) >= HEAVY_CORRECTION or (ax.heavy_since and now - ax.heavy_since >= SUSTAINED_S):
                        self._alert(f"heavy_{a}", f"HEAVY {a.upper()} CORRECTION {total:+.0f}%: a strong draft, or "
                                                  f"the drone turning by itself (land, press C to calibrate)", "warn", now)
                    elif abs(total) < SUSTAINED_CORRECTION - 5:
                        self._clear(f"heavy_{a}")
            reason = None
        else:
            self.hold = {a: 0.0 for a in self.hold}     # never bias a move or the pilot
            for ax in self.axes.values():
                ax.push_since, ax.sign, ax.strong_since = None, 0, None
        if backing is not None:
            when = f"{backing['ttc']:.1f}s" if backing.get("ttc") is not None else "very close"
            self._alert("backoff", f"BACKING AWAY from {backing['label']} #{backing['id']} ({when})", "act", now)
        else:
            self._clear("backoff")
        label = state if reason is None else f"{state}: {reason}"
        if label != self.hold_state:
            flightlog.event("hold", state=state, reason=reason)
        self.hold_state = label

    def _self_check(self, a, ax, now, v):
        name = "sideways" if a == "x" else "forward/back"
        push = self.hold[a]
        sign = 0 if abs(push) < ax.min * 0.8 else (1 if push > 0 else -1)
        # wobbling: the slide keeps reversing while it pushes (it overshoots back and forth)
        vdir = 0 if abs(v) < 2 * HOLD_DEADBAND else (1 if v > 0 else -1)
        if vdir and ax.last_dir and vdir != ax.last_dir:
            ax.flips.append(now)
        if vdir:
            ax.last_dir = vdir
        while ax.flips and now - ax.flips[0] > OSC_WINDOW:
            ax.flips.popleft()
        if len(ax.flips) >= OSC_FLIPS:
            ax.gain = max(ax.gain0 * GAIN_FLOOR, ax.gain * OSC_CUT)
            ax.min = max(MIN_FLOOR, ax.min * OSC_CUT)
            ax.flips.clear()
            flightlog.event("hold_tune", axis=a, gain=round(ax.gain, 1), min=round(ax.min, 1), why="wobbling")
            self._alert(f"tune_{a}", f"HOLD {a.upper()} was wobbling: softened to gain {ax.gain:.0f}, "
                                     f"min {ax.min:.0f}%", "warn", now)
        # wrong direction: pushing against the slide, but the slide keeps growing
        if sign != ax.sign:
            ax.sign, ax.push_since, ax.v_start = sign, (now if sign else None), abs(v)
        elif sign and now - ax.push_since >= WORSE_WINDOW:
            against = (v > 0) == (sign < 0)                # pushing the opposite way to the slide
            if against and abs(v) > WORSE_MIN_V and abs(v) > WORSE_FACTOR * max(ax.v_start, HOLD_DEADBAND):
                restored = None
                if self.autotrim is not None:
                    self.autotrim.skip.add(a)                 # auto-trim reads the same estimate
                if self.pilot:                                # undo this flight's trim on that axis
                    restored = self.takeoff_trim.get(a, 0.0)
                    self.pilot.set_trims({a: restored})
                ax.i = 0.0
                self.hold[a] = 0.0
                ax.sign, ax.push_since, ax.strong_since, ax.cap = 0, None, None, self.hold_max
                ax.flips.clear()
                if not ax.flipped:
                    ax.flipped = True
                    flightlog.event("hold_axis_flip", axis=a, v_start=round(ax.v_start, 4), v_now=round(v, 4),
                                    trim_restored_to=restored)
                    self._alert(f"axis_{a}", f"HOLD {a.upper()} FLIPPED: pushing made the {name} slide worse, "
                                             f"so the camera estimate is reversed on this drone", "warn", now)
                else:
                    ax.enabled = False
                    flightlog.event("hold_axis_off", axis=a, v_start=round(ax.v_start, 4), v_now=round(v, 4),
                                    trim_restored_to=restored)
                    self._alert(f"axis_{a}", f"HOLD {a.upper()} OFF: pushing made the {name} slide worse "
                                             f"both ways", "warn", now)
                return
            ax.push_since, ax.v_start = now, abs(v)       # judge again over the next window
        # at full push and still sliding: raise the cap; at the highest cap and sliding hard: land
        at_cap = abs(push) >= ax.cap - 0.5
        if not (at_cap and abs(v) > STILL_SLIDING) and abs(v) <= RUNAWAY_V:
            if ax.strong_since is not None:
                self._clear(f"drift_{a}")
            ax.strong_since = None
            return
        ax.strong_since = ax.strong_since or now
        if at_cap and ax.cap < HOLD_CAP_LIMIT and now - max(ax.strong_since, ax.raised_at) >= RUNAWAY_STEP_S:
            ax.cap = min(HOLD_CAP_LIMIT, ax.cap + HOLD_CAP_STEP)
            ax.raised_at = now
            flightlog.event("hold_escalate", axis=a, cap=ax.cap, v=round(v, 4))
            self._alert(f"drift_{a}", f"STRONG {name.upper()} DRIFT: hold raised to {ax.cap:.0f}%", "warn", now)
        elif ax.cap >= HOLD_CAP_LIMIT and abs(v) > RUNAWAY_V and now - max(ax.strong_since, ax.raised_at) >= RUNAWAY_LAND_S:
            self._land(f"can't stop a strong {name} drift even at {ax.cap:.0f}% push", now)

    def _unfixable(self, a, ax, now, v):
        """A switched-off axis can't push back: land if it drifts hard for long."""
        if abs(v) <= RUNAWAY_V:
            ax.strong_since = None
            return
        ax.strong_since = ax.strong_since or now
        if now - ax.strong_since >= UNFIXABLE_LAND_S:
            self._land(f"strong {'sideways' if a == 'x' else 'forward/back'} drift and hold is off for it", now)

    # ---------------- LLM, log + HUD ----------------
    def status(self):
        """Safety state for the LLM's get_status."""
        return {"safety_alerts": [t for t, _, _ in self.alerts.values()],
                "hold_axes": {a: {"on": ax.enabled, "flipped": ax.flipped, "gain": round(ax.gain), "max_push": ax.cap}
                              for a, ax in self.axes.items()},
                "auto_landing": self.landing}

    def tel_fields(self, now):
        """Supervisor and plan state for the 5 Hz telemetry line."""
        prog = self.bridge.progress(now)
        return {"hold": {a: round(v, 1) for a, v in self.hold.items()}, "hold_state": self.hold_state,
                "guard": self.guard_state, "alerts": sorted(self.alerts),
                "axes": {a: [ax.enabled, ax.flipped, round(ax.gain), ax.cap, round(ax.i, 1)]
                         for a, ax in self.axes.items()},
                "plan": None if prog is None else {k: prog[k] for k in ("step", "steps", "what", "left")}}

    def hud(self, now, agent):
        prog = self.bridge.progress(now)
        ended = self.bridge.ended
        recent = self.bridge.result and now - self.bridge.ended_t <= 8
        return {
            "model": agent.llm.last_model or (agent.llm.chain[0] if agent.llm.chain else "?"),
            "phase": agent.phase, "phase_s": now - agent.phase_since, "tool": agent.current_tool,
            "command": agent.last_command, "reply": agent.last_reply,
            "plan": prog, "ended": ended,
            "result": None if prog or not recent else self.bridge.result,
            "guard": (self.guard_state, self.guard_detail),
            "hold": (self.hold_state, dict(self.hold)),
            "axes": {a: (ax.enabled, ax.cap) for a, ax in self.axes.items()},
            "alerts": [(t, lvl) for t, lvl, _ in self.alerts.values()],
            "landing": self.landing,
            "log": os.path.basename(flightlog.LOG.path) if flightlog.LOG.path else None,
        }


def _in_path(o):
    x0, _, x1, _ = o["box"]
    return x1 > 0.3 and x0 < 0.7          # obstacles.CORRIDOR: the middle 40% of the picture


class DepthTrust:
    """How much the camera-based distances deserve to be believed right now - never "high".

    info() gives what's known about the depth setup: available, lens_guessed, cal_samples, scale,
    cal_age_days, status. On top of that, the distance ahead is watched while the drone hovers still:
    a reading that jumps around then is a sign the estimate is shaky in this scene."""

    def __init__(self, info):
        self.info = info
        self.samples = deque()                  # (time, metres ahead) while hovering still

    def update(self, now, ahead_m, steady):
        if ahead_m is None or not steady:
            self.samples.clear()
            return
        self.samples.append((now, ahead_m))
        while self.samples and now - self.samples[0][0] > 3.0:
            self.samples.popleft()

    def jitter(self):
        """Relative spread of the distance ahead over the last few seconds of steady hovering, or None."""
        if len(self.samples) < 10 or self.samples[-1][0] - self.samples[0][0] < 1.5:
            return None
        v = sorted(m for _, m in self.samples)
        mid = v[len(v) // 2]
        return (v[int(len(v) * 0.9)] - v[int(len(v) * 0.1)]) / max(mid, 0.1)

    def status(self, ahead):
        info = self.info() or {}
        if not info.get("available"):
            return {"depth": {"available": False, "why": info.get("status", "no depth model")}}
        why = []
        if info.get("lens_guessed"):
            why.append("the lens field of view is a guess (no camera.json): distances can be off by 30% or more")
        if not info.get("cal_samples"):
            why.append("never tape-calibrated (M key against a wall at a known distance)")
        elif not 0.6 <= info.get("scale", 1.0) <= 1.6:
            why.append(f"the calibration applies a large correction (x{info['scale']:.2f}): worth re-checking")
        if info.get("cal_age_days") is not None and info["cal_age_days"] > DEPTH_OLD_DAYS:
            why.append(f"calibrated {info['cal_age_days']:.0f} days ago")
        j = self.jitter()
        if j is not None and j > DEPTH_UNSTEADY:
            why.append(f"the distance ahead jumps about {j * 100:.0f}% while hovering still: unreliable in this scene")
        dist, close = ahead if ahead else (None, False)
        out = {"available": True, "trust": "low" if why else "medium",
               "why": why or ["calibrated and steady, but still a one-camera estimate: never exact"],
               "ahead_m": None if dist is None else round(dist, 2),
               "ahead_m_cautious": None if dist is None else round(dist * 0.8, 2),
               "ahead_close": bool(close)}
        if j is not None:
            out["ahead_jitter_pct"] = round(j * 100)
        return {"depth": out}

