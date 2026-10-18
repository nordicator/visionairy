"""What the calibration hop (preflight.py --fly) learned about this drone, applied to every flight.

calibration.json holds:
  invert          stick axes ("x" right, "y" forward, "yaw" turn) this drone's firmware reverses.
                  Applied in drone.Pilot.sticks(), so keyboard, AI, trim and hold all mean the same
                  direction afterwards.
  estimate_flip   axes whose camera slide estimate (drift.py) comes out reversed. Applied to the
                  estimate before auto-trim, position hold, the HUD and the AI see it.
  latency_s       rough video lag measured from the hop (information; logged).
  altitude        "holds" / "climbs" / "sinks" at centred throttle.
  measured        when.
Speeds you measured go into ai_notes.md instead, where the LLM reads them.
"""
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_FILE = os.path.join(HERE, "calibration.json")
NOTES_FILE = os.path.join(HERE, "ai_notes.md")
NOTES_MARK = "## Measured by preflight.py"


def default():
    return {"invert": {"x": False, "y": False, "yaw": False},
            "estimate_flip": {"x": False, "y": False, "z": False},
            "latency_s": None, "altitude": None, "measured": None}


def load(path=CAL_FILE):
    cal = default()
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return cal
    for key in ("invert", "estimate_flip"):
        cal[key].update({a: bool(v) for a, v in (saved.get(key) or {}).items() if a in cal[key]})
    for key in ("latency_s", "altitude", "measured"):
        cal[key] = saved.get(key, cal[key])
    return cal


def save(cal, path=CAL_FILE):
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)


def apply_est(vel, cal):
    """The drift estimate with this drone's reversed axes put right."""
    flip = cal["estimate_flip"]
    return {a: -v if flip.get(a) else v for a, v in vel.items()}


def describe(cal):
    parts = [f"{a} reversed" for a, on in cal["invert"].items() if on]
    parts += [f"camera {a} estimate reversed" for a, on in cal["estimate_flip"].items() if on]
    if cal["altitude"] and cal["altitude"] != "holds":
        parts.append(f"altitude {cal['altitude']}")
    return ", ".join(parts) or "nothing reversed"


def write_notes(lines, path=NOTES_FILE):
    """Replace the preflight section of ai_notes.md (the LLM reads this file) with these lines."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        text = ""
    text = re.sub(rf"\n*{re.escape(NOTES_MARK)}.*?(?=\n## |\Z)", "", text, flags=re.S).rstrip()
    block = f"{NOTES_MARK} ({time.strftime('%Y-%m-%d %H:%M')})\n" + "\n".join(f"- {ln}" for ln in lines)
    with open(path, "w") as f:
        f.write((text + "\n\n" if text else "") + block + "\n")


class Calibrated:
    """Wraps a drift.DriftEstimator so .vel comes out with this drone's reversed axes put right."""

    def __init__(self, est, cal):
        self.est, self.cal = est, cal

    def update(self, frame, t):
        self.est.update(frame, t)

    @property
    def vel(self):
        return apply_est(self.est.vel, self.cal)

    @property
    def valid(self):
        return self.est.valid

    @valid.setter
    def valid(self, v):
        self.est.valid = v

    @property
    def points(self):
        return self.est.points
