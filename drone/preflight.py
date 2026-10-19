#!/usr/bin/env python3
"""Pre-flight checks and a calibration hop, so the first real AI flight goes smoothly.

  python preflight.py              # checks only, nothing flies (drone link, video, camera texture,
                                   #   internet + LLM tool calling while on the drone's Wi-Fi, trim, setup)
  python preflight.py --fly        # + calibration hop: take off, nudge right / forward / turn, ask you
                                   #   what you saw, land; saves calibration.json and your measurements
  python preflight.py --sim --fly  # rehearse the hop on the simulator (answers itself)
  python preflight.py --sim --fly --invert x --sign-bug --latency 0.3   # ... with a misbehaving drone

The hop needs about 1.5 m clear on every side and a flat floor. Ctrl-C at any point lands it.
For each nudge it asks you which way the drone really went and compares that with what the camera
measured, which tells apart the two things that can be reversed on a new drone:
  - the stick direction (the firmware's convention) -> fixed in the sticks for keyboard and AI
  - the camera's slide estimate                       -> fixed before auto-trim / hold / AI read it
It also measures the video lag, checks the altitude hold, and (if you give rough distances) writes
real speeds into ai_notes.md for the LLM.
"""
import argparse
import json
import math
import os
import select
import sys
import threading
import time

import calibration
import drift
import drone
import flightlog
from protocol import FLAG_GYRO_CAL, FLAG_LAND, FLAG_TAKEOFF

NUDGE = 30.0            # % stick for each test move
NUDGE_S = 0.6           # s of stick
WATCH_S = 1.6           # s of the camera trace analysed from the start of a nudge
SETTLE_S = 1.2          # s of hover between moves
MIN_PEAK = 0.008        # frame-widths/s: smaller camera response = "couldn't see it"
EST_LAG = 0.21          # s the estimator's smoothing takes to reach 30% of a step (tau 0.6)
SPINUP_LAG = 0.15       # s a toy drone takes to start moving after the stick moves
TESTS = [  # (axis, stick level key, question, answer meaning "reversed", camera axis)
    ("x", "right", "Did it first slide RIGHT (then come back)?", "it went LEFT first", "x"),
    ("y", "forward", "Did it first go FORWARD, nose-first (then come back)?", "it went BACK first", "y"),
    ("yaw", "turn", "Did it first turn RIGHT, clockwise seen from above (then back)?", "it turned LEFT first", "x"),
]


# ---------------- backends: the real drone or the simulator ----------------
class RealBackend:
    def __init__(self, fly):
        variant = drone.detect_variant()
        if not variant:
            raise RuntimeError("not on the drone's Wi-Fi (expected a 192.168.1.x or 192.168.169.x address)")
        self.variant = variant
        self.link = drone.Link(variant, fly, None, False, objects=False, model=None)
        self.pilot = self.link.pilot
        self.est = drift.DriftEstimator("forward")
        self.frame_times = []
        self.last_frame = None
        self.resolution = None
        self.lock = threading.Lock()
        threads = [threading.Thread(target=self.link.run_a if variant == "a" else self.link.run_b, daemon=True),
                   threading.Thread(target=self._decode, daemon=True)]
        if variant == "a":
            threads.append(threading.Thread(target=self.link.video_a, daemon=True))
        for t in threads:
            t.start()

    def _decode(self):
        import cv2
        import numpy as np
        shown = 0
        while self.link.running:
            if self.link.frames != shown and self.link.latest_jpeg:
                shown = self.link.frames
                img = cv2.imdecode(np.frombuffer(self.link.latest_jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    now = time.time()
                    with self.lock:
                        self.est.update(img, now)
                        self.frame_times.append(now)
                        self.last_frame = now
                        self.resolution = f"{img.shape[1]}x{img.shape[0]}"
            time.sleep(0.005)

    def sample(self):
        with self.lock:
            age = None if self.last_frame is None else time.time() - self.last_frame
            return dict(self.est.vel), self.est.valid and age is not None and age < 0.5, self.est.points, age

    def truth(self):
        return None

    def close(self):
        self.link.running = False


class SimBackend:
    def __init__(self, invert=(), sign_bug=False, latency=0.0, wind=0.0, alt_drift=0.0, seed=1):
        import sim_flight
        self.variant = "sim"
        self.pilot = drone.Pilot()
        self.lock = threading.Lock()
        self.sim = sim_flight.SimDrone(wind, seed, invert=invert, alt_drift=alt_drift)
        self.camera = sim_flight.SimCamera(self.sim, self.lock, [], sign_bug, latency=latency, seed=seed)
        self.frame_times = []
        self.resolution = "sim"
        self.last = ({"x": 0.0, "y": 0.0, "z": 0.0}, False)
        self.running = True
        threading.Thread(target=self._physics, daemon=True).start()
        threading.Thread(target=self._video, daemon=True).start()

    def _physics(self):
        while self.running:
            s = self.pilot.sticks()
            with self.lock:
                self.sim.step(s, time.time())
            time.sleep(0.005)

    def _video(self):
        while self.running:
            vel, valid, _ = self.camera(time.time())
            self.last = (vel, valid)
            self.frame_times.append(time.time())
            time.sleep(0.04)

    def sample(self):
        vel, valid = self.last
        return dict(vel), valid or self.sim.mode == "ground", 80, 0.04

    def truth(self):
        with self.lock:
            s = self.sim
            return {"x": s.x, "y": s.y, "z": s.z, "heading": s.heading}

    def close(self):
        self.running = False


# ---------------- talking to the person ----------------
def ask(question, auto=None, timeout=25.0):
    """A one-line answer from the terminal (blank after timeout); auto answers instead in the simulator."""
    if auto is not None:
        print(f"  ? {question}  -> {auto or '(blank)'}  [simulator]", flush=True)
        return auto
    print(f"  ? {question} ", end="", flush=True)
    if sys.platform != "win32":
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            print("(no answer)", flush=True)
            return ""
    return sys.stdin.readline().strip().lower()


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail):
        self.rows.append((status, name, detail))
        mark = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "INFO": "    "}[status]
        print(f"  {mark}  {name:22s} {detail}", flush=True)
        flightlog.event("preflight", status=status, check=name, detail=detail)

    def failed(self):
        return any(s == "FAIL" for s, _, _ in self.rows)


# ---------------- checks that don't fly ----------------
def check_video(backend, rep, seconds=5.0):
    t0 = time.time()
    valid = points = n = 0
    while time.time() - t0 < seconds:
        _, ok, pts, _ = backend.sample()
        valid += ok
        points += pts or 0
        n += 1
        time.sleep(0.05)
    times = [t for t in backend.frame_times if t >= t0]
    if len(times) < 3:
        rep.add("FAIL", "video", "no video frames - is the drone on, and are you on its Wi-Fi?")
        return False
    fps = (len(times) - 1) / (times[-1] - times[0])
    gap = max(b - a for a, b in zip(times, times[1:]))
    status = "PASS" if fps >= 12 and gap < 0.4 else "WARN"
    rep.add(status, "video", f"{fps:.0f} fps, {backend.resolution}, longest gap {gap * 1000:.0f} ms"
            + ("" if status == "PASS" else "  (choppy: move closer to the drone, fewer Wi-Fi networks around)"))
    share = valid / max(n, 1)
    status = "PASS" if share >= 0.8 else "WARN" if share >= 0.4 else "FAIL"
    rep.add(status, "camera texture", f"slide estimate usable {share:.0%} of the time, ~{points / max(n, 1):.0f} "
            f"features" + ("" if status == "PASS" else "  (point it at a textured scene: furniture, posters; "
                                                       "not a blank wall; turn the lights on)"))
    return True


def check_llm(rep):
    import ai_pilot
    try:
        llm = ai_pilot.LLM()
    except SystemExit as e:
        rep.add("FAIL", "LLM settings", str(e))
        return
    llm.log = lambda *a: None
    msgs = [{"role": "system", "content": ai_pilot.system_prompt(60, True)},
            {"role": "user", "content": "take off, hover for 1 second, land"}]
    t0 = time.time()
    try:
        msg = llm.chat(msgs, ai_pilot.tool_specs(60, True))
    except Exception as e:
        hint = ""
        if "can't reach" in str(e) or "timed out" in str(e):
            hint = ("  -> no internet while on the drone's Wi-Fi: plug in Ethernet or USB-tether your phone, and put "
                    "that above Wi-Fi in System Settings > Network > ... > Set Service Order")
        rep.add("FAIL", "LLM + internet", f"{str(e)[:120]}{hint}")
        return
    secs = time.time() - t0
    calls = [c["function"]["name"] for c in msg.get("tool_calls") or []]
    if not calls:
        rep.add("FAIL", "LLM tool calling", f"{llm.last_model} answered in text, not tool calls: pick another model")
    else:
        rep.add("PASS" if secs < 10 else "WARN", "LLM + internet",
                f"{llm.last_model} planned {calls} in {secs:.1f} s" + ("" if secs < 10 else "  (slow)"))


def check_setup(rep):
    trim = drift.load_trim(drone.TRIM_FILE)
    big = {a: v for a, v in trim.items() if abs(v) > 15}
    rep.add("WARN" if big else "PASS", "saved trim",
            f"{', '.join(f'{a} {v:+.0f}%' for a, v in trim.items())}"
            + ("  (large: if the drone was changed, reset with BACKSPACE in drone.py)" if big else ""))
    cal = calibration.load()
    if cal["measured"]:
        rep.add("PASS", "calibration", f"from {cal['measured']}: {calibration.describe(cal)}")
    else:
        rep.add("WARN", "calibration", "none yet: run  python preflight.py --fly")
    notes = os.path.exists(calibration.NOTES_FILE)
    rep.add("PASS" if notes else "INFO", "speed notes", "ai_notes.md present" if notes else
            "no ai_notes.md: distances use rough default speeds (the hop can write it)")


# ---------------- the calibration hop ----------------
def record(backend, seconds, levels_at=None):
    """Sample the camera estimate for this long, setting the sticks from levels_at(t) as it goes."""
    out = []
    t0 = time.time()
    while (t := time.time() - t0) < seconds:
        if levels_at:
            backend.pilot.steer(levels_at(t))
        vel, ok, _, _ = backend.sample()
        out.append((t, vel, ok))
        time.sleep(0.02)
    return out


def zero():
    return {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}


def analyse(trace, cam_axis):
    """Peak response of the camera on cam_axis, its sign, and when it started (rough lag)."""
    ok = [(t, v[cam_axis]) for t, v, good in trace if good]
    if len(ok) < 5:
        return None
    base = sum(v for t, v in ok if t < 0.1) / max(1, sum(1 for t, _ in ok if t < 0.1)) if ok[0][0] < 0.1 else 0.0
    rel = [(t, v - base) for t, v in ok if t <= WATCH_S]
    t_peak, peak = max(rel, key=lambda r: abs(r[1]))
    onset = next((t for t, v in rel if abs(v) >= 0.3 * abs(peak) and v * peak > 0), None)
    return {"peak": round(peak, 4), "sign": 0 if abs(peak) < MIN_PEAK else (1 if peak > 0 else -1),
            "onset": onset, "lag": None if onset is None else max(0.0, onset - EST_LAG - SPINUP_LAG)}


def hop(backend, sim):
    pilot = backend.pilot
    pilot.invert = {"x": False, "y": False, "yaw": False}       # measure the raw drone
    if not sim:
        pilot.set_trims(drift.load_trim(drone.TRIM_FILE))       # your saved trim (the simulator needs none)
    found = {"answers": {}, "camera": {}, "distance_cm": {}}

    def say(msg):
        print(f"  > {msg}", flush=True)
        flightlog.event("preflight_step", msg=msg)

    say("calibrating the gyro (drone flat and still)")
    pilot.pulse(FLAG_GYRO_CAL)
    record(backend, 2.6, lambda t: zero())
    say("taking off")
    pilot.pulse(FLAG_TAKEOFF)
    record(backend, 4.0, lambda t: zero())

    say("hovering 4 s: checking altitude hold")
    before = backend.truth()
    trace = record(backend, 4.0, lambda t: zero())
    after = backend.truth()
    zs = [v["z"] for _, v, ok in trace if ok]
    found["camera"]["hover_z"] = round(sum(zs) / len(zs), 4) if zs else None
    auto = None
    if sim:
        dz = after["z"] - before["z"]
        auto = "" if abs(dz) < 0.08 else ("u" if dz > 0 else "d")
    found["answers"]["altitude"] = ask("Is it holding its height? [Enter] yes   u = climbing   d = sinking", auto)

    for axis, key, question, reversed_text, cam in TESTS:
        level = NUDGE + (10 if axis == "yaw" else 0)
        say(f"nudging {key} {level:.0f}% for {NUDGE_S} s, then back")
        start = backend.truth()
        stick = "yaw" if axis == "yaw" else axis

        def first(t, stick=stick, level=level):
            lv = zero()
            if t < NUDGE_S:
                lv[stick] = level
            return lv
        trace = record(backend, WATCH_S, first)
        mid = backend.truth()
        record(backend, SETTLE_S - (WATCH_S - NUDGE_S) if SETTLE_S > WATCH_S - NUDGE_S else 0.2, lambda t: zero())

        def back(t, stick=stick, level=level):
            lv = zero()
            if t < NUDGE_S:
                lv[stick] = -level
            return lv
        record(backend, NUDGE_S + SETTLE_S, back)
        found["camera"][axis] = analyse(trace, cam)
        auto = None
        if sim:
            h = math.radians(start["heading"])
            dx, dy = mid["x"] - start["x"], mid["y"] - start["y"]
            moved = {"x": dx * math.cos(h) - dy * math.sin(h), "y": dx * math.sin(h) + dy * math.cos(h),
                     "yaw": mid["heading"] - start["heading"]}[axis]
            auto = "y" if moved > 0 else "n"
        ans = ask(f"{question}  [y] yes   [n] {reversed_text}   [Enter] not sure", auto)
        found["answers"][axis] = ans
        if axis in ("x", "y") and ans in ("y", "n"):
            auto_cm = None
            if sim:
                auto_cm = str(round(abs(moved) * 100))
            cm = ask("Roughly how far did it go before coming back, in cm? [Enter] skip", auto_cm)
            try:
                found["distance_cm"][axis] = float(cm)
            except ValueError:
                pass
    say("landing")
    pilot.pulse(FLAG_LAND)
    record(backend, 3.5, lambda t: zero())
    return found


def conclude(found, rep, previous):
    cal = calibration.default()
    cal["estimate_flip"] = dict(previous["estimate_flip"])
    lags = []
    for axis, key, _, _, cam in TESTS:
        ans = found["answers"].get(axis, "")
        camera = found["camera"].get(axis)
        if ans not in ("y", "n"):
            cal["invert"][axis] = previous["invert"][axis]
            rep.add("WARN", f"{key} direction", "not sure - kept the previous setting")
            continue
        cal["invert"][axis] = ans == "n"
        rep.add("PASS", f"{key} direction", "REVERSED on this drone - fixed in the sticks" if ans == "n" else "as expected")
        if camera and camera["lag"] is not None:
            lags.append(camera["lag"])
        if axis == "yaw":
            continue
        if not camera or camera["sign"] == 0:
            rep.add("WARN", f"camera {key}", "the camera didn't see the nudge clearly (low texture?) - not changed")
            continue
        physical = -1 if ans == "n" else 1
        cal["estimate_flip"][cam] = camera["sign"] != physical
        rep.add("PASS", f"camera {key}", "estimate REVERSED - fixed for trim/hold" if cal["estimate_flip"][cam]
                else f"estimate matches ({camera['peak']:+.3f} frame-widths/s peak)")
    alt = found["answers"].get("altitude", "")
    cal["altitude"] = {"": "holds", "u": "climbs", "d": "sinks"}.get(alt, "holds")
    rep.add("PASS" if cal["altitude"] == "holds" else "WARN", "altitude hold",
            "holds its height" if cal["altitude"] == "holds" else
            f"{cal['altitude']} at centred throttle - tell Claude; it needs a height correction")
    if lags:
        cal["latency_s"] = round(sorted(lags)[len(lags) // 2], 2)
        rep.add("PASS" if cal["latency_s"] < 0.4 else "WARN", "video lag",
                f"about {cal['latency_s']:.2f} s (rough)" + ("" if cal["latency_s"] < 0.4 else
                                                             "  - the hold will soften itself if it wobbles"))
    cal["measured"] = time.strftime("%Y-%m-%d %H:%M")
    notes = []
    for axis, cm in found["distance_cm"].items():
        name = {"x": "right/left", "y": "forward/back"}[axis]
        notes.append(f"{name} {NUDGE:.0f}% held for {NUDGE_S} s moved about {cm:.0f} cm in total "
                     f"(~{cm / 100 / NUDGE_S:.2f} m/s at {NUDGE:.0f}%: distance = that speed x the time held)")
    return cal, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fly", action="store_true", help="also do the calibration hop")
    ap.add_argument("--no-llm", action="store_true", help="skip the internet + LLM check")
    ap.add_argument("--sim", action="store_true", help="use the simulator instead of the drone")
    ap.add_argument("--invert", default="", help="sim: stick axes the simulated drone reverses, e.g. x,yaw")
    ap.add_argument("--sign-bug", action="store_true", help="sim: camera sideways estimate reversed")
    ap.add_argument("--latency", type=float, default=0.0, help="sim: video lag, s")
    ap.add_argument("--wind", type=float, default=0.0, help="sim: draft, m/s")
    ap.add_argument("--alt-drift", type=float, default=0.0, help="sim: climb/sink at centred throttle, m/s")
    ap.add_argument("--out", help="where to save the calibration (default calibration.json; sim: media/)")
    args = ap.parse_args()
    print("flight log:", os.path.relpath(flightlog.LOG.open("preflight")))
    flightlog.event("session", argv=sys.argv, args=vars(args))
    rep = Report()
    print("\nChecks")
    try:
        backend = (SimBackend([a for a in args.invert.split(",") if a], args.sign_bug, args.latency, args.wind,
                              args.alt_drift) if args.sim else RealBackend(args.fly))
    except RuntimeError as e:
        rep.add("FAIL", "drone network", str(e))
        backend = None
    if backend:
        rep.add("PASS", "drone network", "simulator" if args.sim else f"variant {backend.variant.upper()}")
        check_video(backend, rep)
    if not args.no_llm:
        check_llm(rep)
    check_setup(rep)
    if not args.fly or not backend:
        print("\n" + ("Fix the FAIL lines before flying." if rep.failed() else
                      "Ready. Next: python preflight.py --fly  (calibration hop), then python drone.py --ai --first-flight"))
        if backend:
            backend.close()
        sys.exit(1 if rep.failed() else 0)
    if rep.failed() and not args.sim:
        print("\nNot flying: fix the FAIL lines first.")
        backend.close()
        sys.exit(1)

    print("\nCalibration hop: flat floor, ~1.5 m clear all round, stay out of reach. Ctrl-C lands it.")
    if not args.sim:
        input("  Press Enter to start ")
    try:
        found = hop(backend, args.sim)
    except KeyboardInterrupt:
        print("\n  stopped - landing")
        flightlog.event("preflight_abort")
        backend.pilot.steer(zero())
        backend.pilot.pulse(FLAG_LAND)
        time.sleep(3.5)
        backend.close()
        sys.exit(1)
    flightlog.event("preflight_found", **found)
    print("\nResults")
    cal, notes = conclude(found, rep, calibration.load())
    out = args.out or (os.path.join(drone.HERE, "media", "calibration_sim.json") if args.sim else calibration.CAL_FILE)
    calibration.save(cal, out)
    flightlog.event("calibration_saved", path=out, **cal)
    print(f"\n  saved {os.path.relpath(out)}: {calibration.describe(cal)}")
    for n in notes:
        print(f"  measured: {n}")
    if notes and not args.sim:
        calibration.write_notes(notes)
        print("  wrote the speeds into ai_notes.md (the AI reads it)")
    print("\nNext: python drone.py --ai --first-flight")
    backend.close()
    print("SUMMARY " + json.dumps({"calibration": cal, "notes": notes, "found": found}))


if __name__ == "__main__":
    main()
