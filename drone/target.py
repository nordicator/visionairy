"""Click-to-go: fly the drone towards a clicked spot in the video, keeping its height.

Height (Z) is never touched – throttle stays at hold-altitude – so only X and Y are driven:

  forward camera:  spot right/left of centre -> turn towards it, and fly forward (+Y) while it's
                   roughly ahead, until the spot looks APPROACH_ZOOM times bigger than when clicked
  down camera:     spot right/left of centre -> fly right/left (+X),
                   spot above/below centre   -> fly forward/back (+Y), until it's under the crosshair

The spot is followed from frame to frame with sparse optical flow (features around the click,
median motion), so it stays locked while the drone moves. Because height is held, a spot well
above or below the drone slides off the top/bottom of the forward camera as it closes in; the
drone then stops ("lost"). Strafing barely shifts things far away, so far targets may time out.
"""
import cv2
import numpy as np

WORK_WIDTH = 320          # frames are shrunk to this width before tracking
PATCH = 24                # px (at WORK_WIDTH) around the spot to take features from
MIN_FEATURES = 4          # re-seed features below this
MIN_STICK = 15.0          # % stick when off target at all (toy drones ignore tiny inputs)
GAIN = 45.0               # extra % stick per unit of error (1 = edge of the picture)
MAX_STICK = 50.0          # % stick, cap
FORWARD_STICK = 30.0      # % stick forward while closing in (forward camera)
YAW_GAIN = 55.0           # forward camera: % yaw per unit of error (yaw has no momentum, so no minimum push)
YAW_CAP = 45.0
AIM_CONE = 0.4            # forward camera: flies forward while the spot is within this of centre
APPROACH_ZOOM = 2.0       # forward camera: arrived once the spot looks this many times bigger
ARRIVE_TOL = 0.06         # error (fraction of half the picture) counted as on target
ARRIVE_HOLD = 0.6         # s on target before it counts as arrived
TIMEOUT = 20.0            # s before giving up


class ClickTarget:
    def __init__(self, camera="forward"):
        self.camera = camera
        self.state = "off"      # off | tracking | arrived | lost | timeout
        self.pos = None         # spot in work-res pixels
        self.pending = None     # (u, v) 0..1, waiting for the next frame to lock on
        self.prev = None
        self.pts = self.pts0 = None   # tracked features now / when they were seeded
        self.zoom_base = 1.0          # zoom reached before the last re-seed
        self.zoom = 1.0               # how much bigger the spot looks than when clicked
        self.size = (WORK_WIDTH, 1)
        self.started = self.on_target_since = None

    @property
    def active(self):
        return self.state == "tracking"

    def set(self, u, v, t):
        """Aim at the spot at (u, v), as fractions of the picture width/height."""
        self.pending = (u, v)
        self.pos = None
        self.state = "tracking"
        self.zoom = self.zoom_base = 1.0
        self.started, self.on_target_since = t, None

    def cancel(self, why="off"):
        self.state = why
        self.pending = self.pos = self.pts = self.pts0 = None

    def norm(self):
        """Spot as (u, v) fractions of the picture, or None."""
        if self.pos is None:
            return self.pending
        return self.pos[0] / self.size[0], self.pos[1] / self.size[1]

    def update(self, frame, t):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (WORK_WIDTH, round(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, gray
        if not self.active:
            return
        self.size = (gray.shape[1], gray.shape[0])
        if self.pending is not None:
            self.pos = np.array([self.pending[0] * self.size[0], self.pending[1] * self.size[1]], np.float32)
            self.pending = None
            self._seed(gray)
            return
        if prev is None or prev.shape != gray.shape or self.pts is None:
            self._seed(gray)
            return
        p1, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, self.pts, None, winSize=(21, 21), maxLevel=3)
        ok = ok.ravel() == 1
        if not ok.any():
            return self.cancel("lost")
        moved = (p1[ok] - self.pts[ok]).reshape(-1, 2)
        self.pos = self.pos + np.median(moved, axis=0)
        x, y = self.pos
        if not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            return self.cancel("lost")
        self.pts, self.pts0 = p1[ok].reshape(-1, 1, 2), self.pts0[ok]
        self.zoom = self.zoom_base * self._scale()
        if len(self.pts) < MIN_FEATURES:
            self._seed(gray)

    def _scale(self):
        """How much the patch has grown since it was seeded (1 = same size)."""
        if len(self.pts) < 3:
            return 1.0
        m, _ = cv2.estimateAffinePartial2D(self.pts0, self.pts, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        return 1.0 if m is None else float(np.hypot(m[0, 0], m[1, 0]))

    def _seed(self, gray):
        """Features near the spot, plus the spot itself so even a plain patch gets tracked."""
        self.zoom_base = self.zoom
        mask = np.zeros_like(gray)
        x, y = (int(round(c)) for c in self.pos)
        r = int(PATCH * min(self.zoom, 4.0))             # the patch grows as we close in
        cv2.circle(mask, (x, y), r, 255, -1)
        found = cv2.goodFeaturesToTrack(gray, maxCorners=30, qualityLevel=0.01, minDistance=4, mask=mask)
        spot = self.pos.reshape(1, 1, 2).astype(np.float32)
        self.pts = spot if found is None else np.concatenate([spot, found.astype(np.float32)])
        self.pts0 = self.pts.copy()

    def command(self, t, fresh):
        """Stick levels (% per axis) towards the spot; Z is always 0 (hold altitude)."""
        cmd = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        if not self.active or self.pos is None or not fresh:
            return cmd
        if t - self.started > TIMEOUT:
            self.cancel("timeout")
            return cmd
        u, v = self.norm()
        ex, ey = 2 * u - 1, 2 * v - 1                    # -1..1, + = right / below centre
        if self.camera == "forward":
            # Turn to face it and fly at it. Strafing to centre it (the old way) overshot back and forth
            # and hardly ever moved forward: a sideways move carries on after the stick lets go, a turn doesn't.
            cmd["yaw"] = 0.0 if abs(ex) < ARRIVE_TOL else max(-YAW_CAP, min(YAW_CAP, YAW_GAIN * ex))
            close = self.zoom >= APPROACH_ZOOM
            if not close:
                cmd["y"] = FORWARD_STICK * max(0.0, 1 - abs(ex) / AIM_CONE)
            there = close and abs(ex) < ARRIVE_TOL
        else:
            cmd["x"] = stick(ex)
            cmd["y"] = stick(-ey)
            there = abs(ex) < ARRIVE_TOL and abs(ey) < ARRIVE_TOL
        if there:
            self.on_target_since = self.on_target_since or t
            if t - self.on_target_since >= ARRIVE_HOLD:
                self.cancel("arrived")
                return {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        else:
            self.on_target_since = None
        return cmd


def stick(e):
    if abs(e) < ARRIVE_TOL:
        return 0.0
    return float(np.sign(e)) * min(MAX_STICK, MIN_STICK + GAIN * abs(e))
