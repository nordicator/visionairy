#!/usr/bin/env python3
"""Room-failure test suite: every thing that can go wrong indoors, injected into the simulator.

Each scenario flies a scripted plan (no LLM, no network) in sim_flight.py with one fault switched on
and checks that the supervisor dealt with it. They run in parallel; the whole suite takes ~40 s.

  python test_faults.py            # all
  python test_faults.py wind gale  # some
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
HOVER = [["sequence", {"steps": [{"action": "takeoff"}, {"duration_s": 20}, {"action": "land"}]}]]
FORWARD = [["sequence", {"steps": [{"action": "takeoff"}, {"duration_s": 1}, {"forward": 30, "duration_s": 5},
                                   {"duration_s": 2}, {"action": "land"}]}]]
TOO_FAR = [["sequence", {"steps": [{"action": "takeoff"}, {"forward": 50, "duration_s": 5},
                                   {"forward": 50, "duration_s": 5}, {"action": "land"}]}]]
TOO_HIGH = [["sequence", {"steps": [{"action": "takeoff"}, {"up": 60, "duration_s": 4}, {"action": "land"}]}]]


def ev(d, name, **match):
    return [e for e in d.get("events", []) if e["ev"] == name and all(e.get(k) == v for k, v in match.items())]


def alert(d, key_prefix):
    return [e for e in ev(d, "alert") if (e.get("key") or "").startswith(key_prefix)]


# name: (plan, extra args, check(summary) -> problem or None, what it shows)
SCENARIOS = {
    "calm": (HOVER, [], lambda d: d["end_distance"] < 0.15 or "drifted in still air",
             "still air: stays put"),
    "wind": (HOVER, ["--wind", "0.15"],
             lambda d: (d["end_distance"] < 1.0 and not d["auto_land"]) or f"drifted {d['end_distance']} m",
             "steady draft 0.15 m/s: hold + integral cancel it"),
    "wind_nohold": (HOVER, ["--wind", "0.15", "--no-hold", "--no-autotrim"],
                    lambda d: d["end_distance"] > 2.5 or "the sim's baseline drift changed",
                    "the same draft with nothing correcting (baseline)"),
    "gale": (HOVER, ["--wind", "0.8"],
             # 3.5 m, not 3.0: the hold lets go while landing (it made real landings crooked, see supervisor.py),
             # so a gale carries it further on the way down
             lambda d: d["collisions"] == 0 and (d["auto_land"] or d["end_distance"] < 3.5)
             and ev(d, "hold_escalate") or "didn't escalate or land",
             "strong 0.8 m/s air current: push limit raised until it holds (or it lands)"),
    "spike": (HOVER, ["--spike", "6,0.8,0,1"],
              lambda d: d["end_distance"] < 0.5 or f"didn't recover: {d['end_distance']} m",
              "sudden 1 s shove: brakes and comes back"),
    "reversed_estimate": (HOVER, ["--wind", "0.15", "--wind-dir", "90", "--sign-bug"],
                          lambda d: (d["axes"]["x"]["flipped"] and d["end_distance"] < 2.0)
                          or f"flipped {d['axes']['x']['flipped']}, drift {d['end_distance']} m",
                          "camera's sideways estimate reversed: detects it and flips the axis"),
    "wobble": (HOVER, ["--wind", "0.1", "--hold-gain", "900", "--hold-min", "25"],
               lambda d: d["end_distance"] < 1.2 or f"drifted {d['end_distance']} m",
               "far too aggressive hold settings: still holds (and softens itself if it wobbles)"),
    "yaw_drift": (HOVER, ["--yaw-drift", "8"],
                  lambda d: d["end_distance"] < 0.2 or f"drifted {d['end_distance']} m",
                  "drone turning by itself: the automatic gyro calibration before take off fixes it"),
    "yaw_drift_nogyro": (HOVER, ["--yaw-drift", "8", "--no-auto-gyro"],
                         lambda d: alert(d, "heavy_x") or "no heavy-correction warning",
                         "same without calibrating: can't be fixed in the air, but it warns (known limit)"),
    "video_loss": (HOVER, ["--video-drop", "6,10"],
                   lambda d: "no video" in (d["auto_land"] or "") or f"auto_land={d['auto_land']}",
                   "video gone mid-hover: lands after 5 s"),
    "battery": (HOVER, ["--max-flight", "12"],
                lambda d: "flight time" in (d["auto_land"] or "") or f"auto_land={d['auto_land']}",
                "flight time limit: lands"),
    "drift_into_box": (HOVER, ["--wind", "0.25", "--wind-dir", "0", "--obstacle", "0,1.4"],
                       lambda d: d["collisions"] == 0 or "hit the box",
                       "a draft pushing it toward a box while hovering: backs away"),
    "box_ahead": (FORWARD, ["--obstacle", "0,1.6"],
                  lambda d: (d["collisions"] == 0 and ev(d, "guard_stop")) or "no guard stop / collision",
                  "box in the path of a forward flight: guard stops it"),
    "blank_wall": (FORWARD, ["--blank-wall", "2.0"],
                   lambda d: (d["collisions"] == 0 and ev(d, "guard_stop")) or "hit the blank wall",
                   "featureless wall ahead (the object tracker can't see it): stops on lost texture"),
    "wind_lag": (HOVER, ["--wind", "0.15", "--latency", "0.4"],
                 lambda d: (d["end_distance"] < 1.0 and not d["auto_land"]) or f"drifted {d['end_distance']} m",
                 "the draft again with 0.4 s of video lag (toy drones): still holds"),
    "box_lag": (FORWARD, ["--obstacle", "0,1.6", "--latency", "0.5"],
                lambda d: (d["collisions"] == 0 and ev(d, "guard_stop")) or "hit the box",
                "box ahead with 0.5 s of video lag: guard still stops in time"),
    "depth_sees_wall": (FORWARD, ["--blank-wall", "2.0"],
                        lambda d: (d["collisions"] == 0 and any("camera depth" in (e.get("why") or "")
                                                                for e in ev(d, "guard_stop"))) or "depth didn't stop it",
                        "plain wall ahead: camera depth stops the plan before the texture is even lost"),
    "depth_too_far": (FORWARD, ["--obstacle", "0,1.6", "--depth-scale", "1.8"],
                      lambda d: d["collisions"] == 0 or "hit the box",
                      "depth says things are 80% farther than they are: time-to-contact still stops it"),
    "depth_too_near": (FORWARD, ["--obstacle", "0,1.6", "--depth-scale", "0.5"],
                       lambda d: (d["collisions"] == 0 and ev(d, "guard_stop")) or "no stop",
                       "depth says things are half as far: stops early (the safe way to be wrong)"),
    "no_depth": (FORWARD, ["--obstacle", "0,1.6", "--no-depth"],
                 lambda d: (d["collisions"] == 0 and ev(d, "guard_stop")) or "hit the box",
                 "no depth model at all: the other cues still stop it"),
    "too_far": (TOO_FAR, [], lambda d: not d["flown"] or "flew a plan that leaves the safe area",
                "plan that leaves the room: refused before flying"),
    "too_high": (TOO_HIGH, [], lambda d: not d["flown"] or "flew a plan above the ceiling",
                 "plan that climbs into the ceiling: refused before flying"),
}


def run(name):
    plan, extra, check, _ = SCENARIOS[name]
    t0 = time.time()
    out = subprocess.run([sys.executable, os.path.join(HERE, "sim_flight.py"), "--json", "--plan", json.dumps(plan),
                          "-o", os.path.join(HERE, "media", f"fault_{name}.png"), *extra, name],
                         capture_output=True, text=True, cwd=HERE, timeout=180)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("SUMMARY ")), None)
    if line is None:
        return name, False, f"no summary (exit {out.returncode}): {out.stderr[-300:]}", None, time.time() - t0
    d = json.loads(line[8:])
    try:
        verdict = check(d)
    except (KeyError, TypeError) as e:
        verdict = f"check failed: {e!r}"
    ok = verdict is True or (verdict and not isinstance(verdict, str))
    return name, ok, None if ok else verdict, d, time.time() - t0


def main():
    names = sys.argv[1:] or list(SCENARIOS)
    os.makedirs(os.path.join(HERE, "media"), exist_ok=True)
    with ThreadPoolExecutor(len(names)) as pool:
        results = list(pool.map(run, names))
    failed = 0
    for name, ok, why, d, secs in results:
        desc = SCENARIOS[name][3]
        if d and d.get("flown"):
            info = f"end {d['end_distance']:.2f} m, max {d['max_distance']:.2f} m"
            if d["auto_land"]:
                info += f", landed: {d['auto_land']}"
            if d["collisions"]:
                info += f", {d['collisions']} COLLISION"
        else:
            info = "not flown"
        print(f"{'PASS' if ok else 'FAIL'}  {name:18s} {info:58s} {desc}")
        if not ok:
            failed += 1
            print(f"      -> {why}")
    print(f"\n{len(results) - failed}/{len(results)} passed  (pictures: media/fault_<name>.png, logs: logs/sim_*)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
