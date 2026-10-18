#!/usr/bin/env python3
"""Flight log: one JSON object per line, for debugging a flight afterwards.

Every module calls flightlog.event("name", key=value, ...). Nothing is written until open() is
called (drone.py --ai, ai_pilot.py --dry-run and sim_flight.py do), so tests stay quiet.

Each line has "t" (unix time), "rel" (seconds since the log opened) and "ev" (event name).
Events: session, command, llm_request, llm_response, llm_error, tool, plan_start, plan_step,
plan_end, guard_stop, hold, action, override, toast, error, and "tel" telemetry at 5 Hz
(sticks, drift estimate, hold correction, plan step, objects, link).

  python flightlog.py                    # timeline of the newest log, telemetry left out
  python flightlog.py logs/x.jsonl --tel # with telemetry
  python flightlog.py --tail 40          # last 40 events
"""
import argparse
import glob
import json
import os
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")


class FlightLog:
    def __init__(self):
        self.lock = threading.Lock()
        self.f = None
        self.path = None
        self.t0 = time.time()

    def open(self, prefix="flight"):
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(LOG_DIR, f"{prefix}_{stamp}.jsonl")
        try:
            self.f = open(self.path, "x", buffering=1)  # line-buffered: survives a crash
        except FileExistsError:                         # another run started in the same second
            self.path = os.path.join(LOG_DIR, f"{prefix}_{stamp}_{os.getpid()}.jsonl")
            self.f = open(self.path, "a", buffering=1)
        self.t0 = time.time()
        return self.path

    def event(self, ev, **fields):
        if self.f is None:
            return
        now = time.time()
        rec = {"t": round(now, 3), "rel": round(now - self.t0, 3), "ev": ev, **fields}
        line = json.dumps(rec, default=_jsonable, separators=(",", ":"))
        with self.lock:
            try:
                self.f.write(line + "\n")
            except (OSError, ValueError):
                pass

    def error(self, where, exc=None):
        self.event("error", where=where, error=repr(exc) if exc else None,
                   trace=traceback.format_exc() if exc else None)


def _jsonable(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "item"):          # numpy numbers
        return o.item()
    return repr(o)


LOG = FlightLog()
event = LOG.event
error = LOG.error


# ---------------- reading logs back ----------------
def _fmt(rec):
    ev = rec["ev"]
    skip = {"t", "rel", "ev"}
    body = {k: v for k, v in rec.items() if k not in skip}
    if ev == "tel":
        p = body.get("plan") or {}
        return (f"sticks {body.get('levels')} hold {body.get('hold')} drift {body.get('drift')} "
                f"valid {body.get('drift_valid')} step {p.get('step')}/{p.get('steps')} "
                f"objects {len(body.get('objects') or [])} age {body.get('age_ms')}ms")
    s = json.dumps(body, separators=(", ", ": "))
    return s if len(s) < 400 else s[:400] + " ..."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="log file (default: the newest in logs/)")
    ap.add_argument("--tel", action="store_true", help="include 5 Hz telemetry lines")
    ap.add_argument("--tail", type=int, help="only the last N events")
    args = ap.parse_args()
    path = args.path or max(glob.glob(os.path.join(LOG_DIR, "*.jsonl")), key=os.path.getmtime, default=None)
    if not path:
        raise SystemExit("no logs in logs/")
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if not args.tel:
        recs = [r for r in recs if r["ev"] != "tel"]
    if args.tail:
        recs = recs[-args.tail:]
    print(path)
    for r in recs:
        print(f"{r['rel']:9.2f}  {r['ev']:13s} {_fmt(r)}")


if __name__ == "__main__":
    main()
