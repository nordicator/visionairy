"""The calibration hop rehearsed on simulated drones that are wrong in different ways.

Checks that preflight.py --sim --fly finds exactly what's wrong (and nothing else), then that the
saved calibration makes a reversed drone fly the right way in the simulator.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = {  # name: (sim faults, expected invert, expected estimate_flip, expected altitude)
    "normal": ([], {}, {}, "holds"),
    "roll_and_camera_reversed": (["--invert", "x", "--sign-bug", "--latency", "0.3"], {"x"}, {"x"}, "holds"),
    "pitch_yaw_reversed_climbs": (["--invert", "y,yaw", "--alt-drift", "0.05", "--latency", "0.4"],
                                  {"y", "yaw"}, set(), "climbs"),
    "sinks_in_a_draft": (["--alt-drift", "-0.05", "--wind", "0.1"], set(), set(), "sinks"),
}


def run(name):
    faults, inv, flip, alt = CASES[name]
    out = os.path.join(HERE, "media", f"calibration_{name}.json")
    p = subprocess.run([sys.executable, os.path.join(HERE, "preflight.py"), "--sim", "--fly", "--no-llm",
                        "--out", out, *faults], capture_output=True, text=True, cwd=HERE, timeout=200)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("SUMMARY ")), None)
    if not line:
        return name, f"no summary: {p.stdout[-400:]} {p.stderr[-400:]}"
    cal = json.loads(line[8:])["calibration"]
    got_inv = {a for a, on in cal["invert"].items() if on}
    got_flip = {a for a, on in cal["estimate_flip"].items() if on}
    problems = []
    if got_inv != set(inv):
        problems.append(f"invert {sorted(got_inv)} != {sorted(inv)}")
    if got_flip != set(flip):
        problems.append(f"estimate_flip {sorted(got_flip)} != {sorted(flip)}")
    if cal["altitude"] != alt:
        problems.append(f"altitude {cal['altitude']} != {alt}")
    return name, "; ".join(problems) or None


def fly_right(cal_path):
    plan = [["sequence", {"steps": [{"action": "takeoff"}, {"duration_s": 2}, {"right": 40, "duration_s": 2},
                                    {"duration_s": 4}, {"action": "land"}]}]]
    args = [sys.executable, os.path.join(HERE, "sim_flight.py"), "--json", "--plan", json.dumps(plan), "--invert", "x",
            "--sign-bug", "-o", os.path.join(HERE, "media", "cal_fly_right.png"), "right"]
    if cal_path:
        args[2:2] = ["--calibration", cal_path]
    p = subprocess.run(args, capture_output=True, text=True, cwd=HERE, timeout=200)
    return json.loads(next(ln for ln in p.stdout.splitlines() if ln.startswith("SUMMARY "))[8:])["x"]


def main():
    os.makedirs(os.path.join(HERE, "media"), exist_ok=True)
    with ThreadPoolExecutor(len(CASES)) as pool:
        results = list(pool.map(run, CASES))
    failed = 0
    for name, problem in results:
        print(f"{'PASS' if not problem else 'FAIL'}  calibrate {name}" + (f"  -> {problem}" if problem else ""))
        failed += bool(problem)
    cal = os.path.join(HERE, "media", "calibration_roll_and_camera_reversed.json")
    with ThreadPoolExecutor(2) as pool:
        without, with_cal = pool.map(fly_right, [None, cal])
    ok = without < -0.5 and with_cal > 0.5
    print(f"{'PASS' if ok else 'FAIL'}  'slide right' on a reversed drone: {without:+.2f} m without calibration, "
          f"{with_cal:+.2f} m with it")
    failed += not ok
    print(f"\n{len(results) + 1 - failed}/{len(results) + 1} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
