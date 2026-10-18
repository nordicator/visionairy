"""Drift detection and auto-trim from the drone's video (the drone sends no telemetry).

DriftEstimator tracks features between consecutive frames (sparse optical flow) and fits one
similarity transform per frame pair: image shift + zoom. That is turned into how the drone itself
is moving, in frame-widths per second:

  forward camera:  scene slides left  -> drone moving right (+X)
                   scene zooms in     -> drone moving forward (+Y)
                   scene slides down  -> drone moving up (+Z)
  down camera:     scene slides left  -> right (+X), slides down -> forward (+Y), shrinks -> up (+Z)

AutoTrim watches that estimate while the pilot holds no movement keys and walks a stick offset
(trim) against the drift until it stops, like tapping the app's trim buttons for you. Each time
the pilot lets go (and right after take off) it first calibrates: a few seconds of faster
correction, then it drops back to a slow hold.

Limits: a sideways slide from a forward camera can also be the drone turning (yaw), and a scene
with no texture (blank wall, sky) gives no estimate – the HUD then shows "low texture".
"""
import json
import math
import os

import cv2
import numpy as np

AXES = ("x", "y", "z")
WORK_WIDTH = 320          # frames are shrunk to this width before tracking
MIN_POINTS = 15           # fewer RANSAC inliers than this -> no estimate
SMOOTH_TAU = 0.6          # s, low-pass on the velocity estimate

TRIM_LIMIT = {"x": 25.0, "y": 25.0, "z": 10.0}   # % stick
TRIM_GAIN = 40.0          # % stick per second, per frame-width/s of drift
TRIM_MAX_RATE = 4.0       # % stick per second, fastest the auto-trim may move
DEADBAND = 0.006          # frame-widths/s treated as "not drifting"
SETTLE = 0.7              # s with no movement keys before drift is believed (momentum dies out)
AFTER_TAKEOFF = 2.0       # s after take off before auto-trim starts
CALIBRATE_SECONDS = 3.0   # s of fast correction each time the pilot stops (and after take off)
CALIBRATE_GAIN = 2.5      # gain and max rate are multiplied by this while calibrating


class DriftEstimator:
    def __init__(self, camera="forward"):
        self.camera = camera
        self.prev = None
        self.prev_t = 0.0
        self.vel = dict.fromkeys(AXES, 0.0)   # drone motion, frame-widths per second (smoothed)
        self.points = 0                        # inliers in the last fit
        self.valid = False

    def update(self, frame, t):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (WORK_WIDTH, round(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev, prev_t = self.prev, self.prev_t
        self.prev, self.prev_t = gray, t
        dt = t - prev_t
        if prev is None or prev.shape != gray.shape or not 0.005 < dt < 0.5:
            self.valid = False
            return
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=150, qualityLevel=0.01, minDistance=8)
        if p0 is None or len(p0) < MIN_POINTS:
            return self._miss()
        p1, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
        ok = ok.ravel() == 1
        if ok.sum() < MIN_POINTS:
            return self._miss()
        center = np.array([gray.shape[1] / 2, gray.shape[0] / 2], np.float32)
        a, b = p0[ok].reshape(-1, 2) - center, p1[ok].reshape(-1, 2) - center
        m, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if m is None or inliers is None or inliers.sum() < MIN_POINTS:
            return self._miss()
        self.points = int(inliers.sum())
        gw = gray.shape[1]
        sx = m[0, 2] / gw / dt                              # scene shift, frame-widths/s (+ = right)
        sy = m[1, 2] / gw / dt                              #                              (+ = down)
        zoom = math.log(math.hypot(m[0, 0], m[1, 0])) / dt  # + = scene growing
        if self.camera == "down":
            raw = {"x": -sx, "y": sy, "z": -zoom}
        else:
            raw = {"x": -sx, "y": zoom, "z": sy}
        k = 1 - math.exp(-dt / SMOOTH_TAU)
        for axis in AXES:
            self.vel[axis] += k * (raw[axis] - self.vel[axis])
        self.valid = True

    def _miss(self):
        self.points = 0
        self.valid = False


class AutoTrim:
    """Integral controller: trim -= gain * drift, only while hands-off and the estimate is good."""

    def __init__(self, pilot, enabled=True):
        self.pilot = pilot
        self.enabled = enabled
        self.state = "off"      # off | waiting | calibrating | learning | steady | paused | low texture
        self.skip = set()       # axes not to trim (the supervisor found the estimate reversed there)

    def update(self, est, dt, airborne_for, idle_for):
        """airborne_for: s since take off (None on the ground); idle_for: s since the last movement key."""
        if not self.enabled:
            self.state = "off"
        elif airborne_for is None or airborne_for < AFTER_TAKEOFF:
            self.state = "waiting"          # on the ground, or still climbing out
        elif idle_for < SETTLE:
            self.state = "paused"           # pilot is flying it
        elif not est.valid:
            self.state = "low texture"
        else:
            calibrating = (idle_for < SETTLE + CALIBRATE_SECONDS
                           or airborne_for < AFTER_TAKEOFF + CALIBRATE_SECONDS)
            boost = CALIBRATE_GAIN if calibrating else 1.0
            moving = False
            for axis in AXES:
                if axis in self.skip:
                    continue
                v = est.vel[axis]
                if abs(v) > DEADBAND:
                    lim = TRIM_MAX_RATE * boost
                    rate = max(-lim, min(lim, TRIM_GAIN * boost * v))
                    self.pilot.adjust_trim(axis, -rate * dt)
                    moving = True
            self.state = "calibrating" if calibrating else "learning" if moving else "steady"


def describe(vel):
    """'right + forward' style summary of a drift vector, or 'none'."""
    words = []
    for axis, (neg, pos) in (("y", ("back", "forward")), ("x", ("left", "right")), ("z", ("down", "up"))):
        if abs(vel[axis]) > DEADBAND:
            words.append(pos if vel[axis] > 0 else neg)
    return " + ".join(words) or "none"


def load_trim(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return {a: max(-TRIM_LIMIT[a], min(TRIM_LIMIT[a], float(data.get(a, 0)))) for a in AXES}
    except (OSError, ValueError, AttributeError):
        return dict.fromkeys(AXES, 0.0)


def save_trim(path, trim):
    try:
        with open(path, "w") as f:
            json.dump({a: round(trim[a], 2) for a in AXES}, f)
    except OSError:
        pass
