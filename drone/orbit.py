"""Orbit: circle a person with the camera on them the whole time ("do a 360 around me").

Uses the object tracker's human boxes (obstacles.py), no depth:

  find     no human in view -> turn on the spot until one shows up (gives up after a full turn)
  orbit    locked on one person (by track id):
             sideways  constant strafe (STRAFE %), the direction you circle
             turn      keeps them centred: P on how far off-centre the box is, plus an integral that
                       learns the steady turn rate the circle needs
             forward   keeps the box the height it had at lock-on, so the distance stays the same
           the angle travelled is the drone's heading change: the background sweeping across the picture
           (the drift estimator's sideways reading) times the field of view
  done     after DEGREES (360 by default), or ORBIT_TIMEOUT; sticks centred (hover)
  lost     the person out of sight for LOST_S -> stop and hover

Turning (yaw) is used to point, never strafing, because yaw has no momentum to overshoot.
"""
import math

FIND_TURN = 35.0          # % yaw while searching
FIND_DEGREES = 380.0      # give up searching after about a full turn
STRAFE = 25.0             # % sideways while circling
YAW_P = 60.0              # % yaw per unit of off-centre (1 = edge of the picture)
YAW_I = 25.0              # % yaw per second per unit of off-centre, builds up the circle's turn rate
YAW_MAX = 60.0
FWD_P = 80.0              # % forward per unit of relative size error
FWD_MAX = 25.0
SIZE_DEAD = 0.08          # relative box-height error ignored (boxes jitter)
CENTRE_DEAD = 0.04        # off-centre ignored
LOST_S = 1.5
ORBIT_TIMEOUT = 60.0
FIND_TIMEOUT = 20.0
YAW_RATE_GUESS = 1.8      # deg/s per % yaw, when the video can't measure the turn (sim: 50% ~ 90 deg/s)


class Orbit:
    def __init__(self, hfov_deg=90.0):
        self.hfov = hfov_deg
        self.state = "off"        # off | finding | orbiting | done | lost | not found | timeout
        self.target = None        # track id
        self.h0 = None            # box height at lock-on (fraction of the picture)
        self.turned = 0.0         # deg the drone has turned while orbiting (= degrees round the person)
        self.found_turn = 0.0
        self.i = 0.0
        self.sign = 1             # +1: strafe right (circles anticlockwise seen from above)
        self.degrees = 360.0
        self.started = self.seen_at = self.last_t = None
        self.last_yaw = 0.0

    @property
    def active(self):
        return self.state in ("finding", "orbiting")

    def start(self, now, target_id=None, direction="right", degrees=360.0):
        self.state = "finding"
        self.target = target_id
        self.sign = -1 if direction == "left" else 1
        self.degrees = max(30.0, min(720.0, float(degrees)))
        self.h0, self.turned, self.found_turn, self.i = None, 0.0, 0.0, 0.0
        self.started = self.seen_at = self.last_t = now
        self.last_yaw = 0.0

    def cancel(self, why="off"):
        self.state = why

    def status(self):
        if self.state == "finding":
            return "ORBIT: looking for you"
        if self.state == "orbiting":
            return f"ORBIT: {self.turned:.0f} / {self.degrees:.0f} deg"
        return ""

    def _pick(self, objects):
        people = [o for o in objects or [] if o["label"] == "human" and not o["lost"]]
        if self.target is not None:
            return next((o for o in people if o["id"] == self.target), None)
        # the biggest person in view (closest, most likely the one who asked)
        return max(people, key=lambda o: o["box"][3] - o["box"][1], default=None)

    def _swept(self, dt, est, yaw):
        """deg turned this frame: from the background's sweep when the video has it, else from the stick"""
        if est is not None and est.valid:
            return abs(est.vel["x"]) * self.hfov * dt
        return abs(yaw) * YAW_RATE_GUESS * dt

    def command(self, now, objects, est=None):
        hold = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        if not self.active:
            return hold
        dt = max(0.0, now - self.last_t)
        self.last_t = now
        person = self._pick(objects)

        if self.state == "finding":
            if person is not None:
                x0, y0, x1, y1 = person["box"]
                self.target, self.h0 = person["id"], y1 - y0
                self.state, self.seen_at, self.started = "orbiting", now, now
                return hold
            self.found_turn += self._swept(dt, est, FIND_TURN)
            if self.found_turn > FIND_DEGREES or now - self.started > FIND_TIMEOUT:
                self.cancel("not found")
                return hold
            return {**hold, "yaw": FIND_TURN}

        # orbiting
        if person is None:
            if now - self.seen_at > LOST_S:
                self.cancel("lost")
            # coast on the learned turn rate, no strafe, so it swings back towards where they were
            return {**hold, "yaw": max(-YAW_MAX, min(YAW_MAX, self.i))}
        self.seen_at = now
        x0, y0, x1, y1 = person["box"]
        ex = (x0 + x1) - 1                      # -1..1, + = right of centre
        ex = 0.0 if abs(ex) < CENTRE_DEAD else ex
        self.i = max(-YAW_MAX, min(YAW_MAX, self.i + YAW_I * ex * dt))
        yaw = max(-YAW_MAX, min(YAW_MAX, YAW_P * ex + self.i))
        size_err = (self.h0 - (y1 - y0)) / self.h0   # + = looks smaller = too far
        fwd = 0.0 if abs(size_err) < SIZE_DEAD else max(-FWD_MAX, min(FWD_MAX, FWD_P * size_err))
        # strafe less while they're far off-centre, so turning catches up before they slide out of view
        strafe = self.sign * STRAFE * max(0.0, 1 - 1.5 * abs(ex))
        self.turned += self._swept(dt, est, yaw)
        if self.turned >= self.degrees:
            self.cancel("done")
            return hold
        if now - self.started > ORBIT_TIMEOUT:
            self.cancel("timeout")
            return hold
        return {"x": strafe, "y": fwd, "z": 0.0, "yaw": yaw}
