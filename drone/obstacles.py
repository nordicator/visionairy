"""Find, name and track objects / obstacles in the drone's video.

Names shown: human, chair, table, tv, bag, box, window, door, wall. Anything else is tracked but
only shown while it's in the way, as "obstacle".

Two detectors feed one tracker:
  named      a small YOLOX model (detector.py) on a background thread: human, chair, table, tv,
             bag. It's sure of itself; these are the reliable names.
  generic    the model-free detector below. It finds anything that stands out, and names a few
             things by shape rules - rougher guesses than the model's:
               box     a solid outline with 4-6 corners
               window  a bright, roughly rectangular region (daylight; a lit lamp or a white
                       sheet fools it, a window at night isn't found)
               door    two long upright lines a door's width apart reaching towards the floor,
                       joined by a level line across the top (wardrobes can look the same)
               wall    a big blank area around the centre of the picture
             Generic boxes that land on a named one are dropped, so a human isn't also chopped
             into pieces.

Generic detect (every frame, on a 320 px wide copy, a few ms):
  saliency   spectral-residual saliency on a 64 px thumbnail: regions unlike the rest of the
             picture light up, while repeated texture (carpet, grass, sky) is suppressed
  edges      Canny edges on each colour channel (so a blue thing on a grey floor still shows),
             closed into solid outlines, for objects with clear boundaries
  Both give candidate boxes; too-small / too-big / thin boxes are dropped, and a box lying mostly
  inside a bigger one is treated as a piece of it.

Track (built for a camera that moves a lot):
  Camera motion  each frame, how the whole picture shifted / turned / zoomed is measured with
                 optical flow, and every track's box is moved with it before matching. The model's
                 answers arrive a few frames late, so they're carried forward the same way.
  Blur           frames smeared by a sudden move are not detected on; tracks just ride along.
  Matching       each box goes to the track it overlaps most (IoU, then centre distance). The
                 model must be BIRTH_CONF sure to start a track; weaker hits only keep one alive.
  Names          each track keeps a decaying vote of the names it was given. A name shows once
                 its vote reaches NAME_MIN and only changes when another out-votes it by SWITCH.
                 Once named, a track ignores model boxes with a different name (those are the
                 table behind the chair, not a new opinion about the chair).
  Lifetime       shown after CONFIRM sightings; kept COAST s unseen (COAST_NAMED once named).
                 No new track starts on top of an existing one. Two tracks on the same thing are merged, the older ID wins.

Obstacle warning (forward camera):
  A box that keeps growing is something we're closing in on. Its growth rate gives a
  time-to-contact (TTC = size / growth per second). A track is a "threat" when it sits in the
  flight corridor (the middle of the picture) and is either big (NEAR) or TTC < TTC_WARN.

A door is never a threat - it's the way through.

Limits: low light and heavy JPEG blocking cause flicker; the shape rules are guesses.
"""
import math
from collections import deque

import cv2
import numpy as np

import detector

WORK_WIDTH = 320          # frames are shrunk to this width before detection
SAL_WIDTH = 64            # saliency is computed on a thumbnail this wide
SAL_K = 3.0               # saliency above this many times its mean counts as "stands out"
MIN_AREA = 0.004          # box area, fraction of the picture
MAX_AREA = 0.5
MAX_ASPECT = 6.0          # longer side / shorter side
CANNY = (30, 90)          # edge thresholds, per colour channel
INSIDE = 0.4              # a box with this much of itself inside a bigger box is part of it
MAX_TRACKS = 16

BOX_FILL = 0.8            # outline filling this much of its bounding box, with 4-6 corners = "box"
BOX_MAX_AREA = 0.25
WINDOW_MIN = 150          # window: brighter than this (0-255) ...
WINDOW_ABOVE = 45         # ... and this much above the picture's median brightness
WINDOW_AREA = (0.01, 0.5)
WINDOW_FILL = 0.7
DOOR_MIN_H = 0.35         # door: two upright lines this tall (fraction of picture height) ...
DOOR_RATIO = (0.3, 0.75)  # ... this far apart (width / height) ...
DOOR_FLOOR = 0.55         # ... reaching at least this far down the picture,
DOOR_LINTEL = 0.6         # ... with a level line across this much of the top
WALL_AREA = 0.35          # blank area around the centre this big = wall ahead
WALL_LIGHT = (25, 235)    # mean brightness range in which a blank area can be judged
LONG_LINE = 0.4           # edges longer than this (fraction of the width) are room structure
SCENE = {"wall", "door", "window"}
HEURISTIC_VOTE = 0.2      # a shape-rule name counts this much (model names count their score)

MATCH_IOU = 0.15          # min overlap to match a box to a track
MATCH_DIST = 0.1          # ... or max centre distance, fraction of the picture width
MODEL_WEIGHT = 0.7        # 0..1, how far a model box moves its track (1 = jump straight to it)
SHAPE_WEIGHT = 0.4        # ... and a shape-rule / generic box (they wobble more)
CONFIRM = 3               # times seen before a track is shown
COAST = 0.6               # s a track survives without being seen ...
COAST_NAMED = 1.5         # ... or once it's named (camera-motion tracking keeps it in place)
GROWTH_TAU = 0.5          # s, low-pass on the growth rate
VEL_GAIN = 0.15           # how fast a track learns its own motion (on top of the camera's)
MAX_VEL = 1.0             # picture fractions / s, cap on a track's own motion
MERGE_IOU = 0.7           # two tracks overlapping this much are the same thing
BIRTH_BLOCK = 0.4         # no new track on a box overlapping an existing track this much

BIRTH_CONF = 0.45         # model score needed to start a track (lower scores only keep one going)
NAME_MIN = 0.6            # vote needed before a name is shown (~2 model sightings, ~4 shape ones)
SWITCH = 1.5              # a new name must out-vote the shown one by this factor to replace it
MIN_FLOW_POINTS = 12      # tracked points needed to measure camera motion
MOTION_HISTORY = 30       # frames of camera motion kept, to carry late model results forward
BLUR_DROP = 0.45          # a frame less sharp than this x the recent average is skipped

CORRIDOR = 0.4            # middle fraction of the picture width the drone flies through
NEAR = 0.45               # box this tall (fraction of the picture) = very close
TTC_WARN = 2.0            # s, time-to-contact that counts as a threat
TTC_SHOW = 6.0            # s, time-to-contact worth showing at all
NAMED_FRESH = 0.5         # s a named box keeps suppressing generic boxes on top of it
VOTE_DECAY = 0.85         # older name votes fade by this much per new name


def area(r):
    return (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])


def intersection(a, b):
    """Overlap area of every box in a (N, 4) with every box in b (M, 4), boxes as x0 y0 x1 y1."""
    a, b = a[:, None], b[None]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    return iw * ih


def iou(a, b):
    inter = intersection(a, b)
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def saliency(gray):
    """Spectral-residual saliency (Hou & Zhang 2007), 0..1, at the size of gray."""
    h, w = gray.shape
    th = max(8, round(h * SAL_WIDTH / w))
    small = cv2.resize(gray, (SAL_WIDTH, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    pad = SAL_WIDTH // 8                     # mirrored border: stops the FFT wrapping edges around
    small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    spec = np.fft.fft2(small)
    log_amp = np.log(np.abs(spec) + 1e-6).astype(np.float32)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spec)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2.0)[pad:-pad, pad:-pad]
    return cv2.resize(sal / (sal.max() + 1e-9), (w, h), interpolation=cv2.INTER_LINEAR)


def boxes_from(mask, w, h, shapes=False):
    """Boxes around the blobs in mask. With shapes, solid near-rectangular ones are called "box"."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if MIN_AREA <= bw * bh / (w * h) <= MAX_AREA and max(bw, bh) <= MAX_ASPECT * min(bw, bh):
            label = None
            if shapes and bw * bh / (w * h) <= BOX_MAX_AREA:
                hull = cv2.convexHull(c)                   # forgives small gaps in the outline
                corners = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)
                if cv2.contourArea(hull) >= BOX_FILL * bw * bh and 4 <= len(corners) <= 6:
                    label = "box"
            out.append(((x / w, y / h, (x + bw) / w, (y + bh) / h), label))
    return out


def windows(gray):
    """Bright, roughly rectangular regions: daylight windows."""
    gh, gw = gray.shape
    thresh = max(WINDOW_MIN, float(np.median(gray)) + WINDOW_ABOVE)
    mask = cv2.morphologyEx((gray > thresh).astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    out = []
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, bw, bh = cv2.boundingRect(c)
        # bright pixels must fill the box: a bright frame round something dark isn't a window
        if WINDOW_AREA[0] <= bw * bh / (gw * gh) <= WINDOW_AREA[1] and                 (gray[y:y + bh, x:x + bw] > thresh).mean() >= WINDOW_FILL:
            out.append(((x / gw, y / gh, (x + bw) / gw, (y + bh) / gh), "window"))
    return out


def doors(edges, gray):
    """Two long upright lines about a door's width apart, reaching down towards the floor, with a
    level line (the lintel) across the top."""
    gh, gw = edges.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=0.06 * gw, maxLineGap=6)
    if lines is None:
        return []
    lines = lines.reshape(-1, 4).astype(float)
    ups = sorted(((x0 + x1) / 2, min(y0, y1), max(y0, y1)) for x0, y0, x1, y1 in lines
                 if abs(x1 - x0) < 0.15 * abs(y1 - y0) and abs(y1 - y0) >= DOOR_MIN_H * gh * 0.7)
    levels = [(min(x0, x1), max(x0, x1), (y0 + y1) / 2) for x0, y0, x1, y1 in lines
              if abs(y1 - y0) < 0.1 * abs(x1 - x0)]

    def lintel(xa, xb, top):
        across = sum(max(0.0, min(x1, xb) - max(x0, xa)) for x0, x1, y in levels
                     if abs(y - top) < 0.06 * gh)
        return across >= DOOR_LINTEL * (xb - xa)

    pairs = []
    for i, (xa, ta, ba) in enumerate(ups):
        for xb, tb, bb in ups[i + 1:]:
            top, bottom = max(ta, tb), min(ba, bb)
            tall = bottom - top
            if tall < DOOR_MIN_H * gh or bottom < DOOR_FLOOR * gh:
                continue
            if DOOR_RATIO[0] <= (xb - xa) / tall <= DOOR_RATIO[1] and lintel(xa, xb, top):
                pairs.append((tall, (xa / gw, top / gh, xb / gw, bottom / gh)))
    out = []
    for _, box in sorted(pairs, reverse=True):              # tallest first, no overlapping doors
        if len(out) < 2 and all(iou(np.array([box]), np.array([b]))[0, 0] < 0.3 for b, _ in out):
            x0, y0, x1, y1 = (round(box[0] * gw), round(box[1] * gh), round(box[2] * gw), round(box[3] * gh))
            inside = gray[y0:y1, x0:x1]                     # bright inside = a window frame instead
            bright = inside.size and inside.mean() > max(WINDOW_MIN, np.median(gray) + WINDOW_ABOVE)
            out.append((box, "window" if bright else "door"))
    return out


def wall(edges, gray):
    """A big blank area covering the middle of the picture: a wall (or ceiling) close ahead."""
    gh, gw = edges.shape
    textured = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(255 - textured, connectivity=4)
    lab = labels[gh // 2, gw // 2]
    if lab == 0 or stats[lab, cv2.CC_STAT_AREA] < WALL_AREA * gw * gh:
        return []
    mean = gray[labels == lab].mean()
    if not WALL_LIGHT[0] < mean < WALL_LIGHT[1]:            # pitch dark or blown out: can't tell
        return []
    x, y, bw, bh = stats[lab, :4]
    return [((x / gw, y / gh, (x + bw) / gw, (y + bh) / gh), "wall")]


def without_long_lines(edges):
    """Edges minus room structure (floor / ceiling lines across the view, long uprights), so things
    standing on the floor don't all merge into one outline with it."""
    gh, gw = edges.shape
    out = edges.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=LONG_LINE * gh, maxLineGap=8)
    for x0, y0, x1, y1 in ([] if lines is None else lines.reshape(-1, 4)):
        if abs(x1 - x0) >= LONG_LINE * gw or abs(y1 - y0) >= 0.9 * gh:
            cv2.line(out, (int(x0), int(y0)), (int(x1), int(y1)), 0, 3)
    return out


def detect(frame):
    """Candidate boxes (N, 4 array of x0 y0 x1 y1 fractions of the picture) and their labels:
    "box" / "window" / "door" / "wall", or None for an unnamed thing that stands out."""
    return _detect(shrink(frame))


def shrink(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (WORK_WIDTH, round(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)


def _detect(small):
    small = cv2.GaussianBlur(small, (5, 5), 0)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    sal = saliency(gray)
    sal_mask = (sal > SAL_K * sal.mean()).astype(np.uint8) * 255
    # big plain objects only light up along their outline: close the ring into one blob
    sal_mask = cv2.morphologyEx(sal_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    raw_edges = np.bitwise_or.reduce([cv2.Canny(c, *CANNY) for c in cv2.split(small)])
    edges = cv2.morphologyEx(without_long_lines(raw_edges), cv2.MORPH_CLOSE, kernel, iterations=2)

    # things: a box lying mostly inside a bigger one is a piece of it (biggest first)
    things = sorted(boxes_from(sal_mask, gw, gh) + boxes_from(edges, gw, gh, shapes=True),
                    key=lambda d: -area(np.array(d[0])))
    boxes = np.array([b for b, _ in things], np.float32).reshape(-1, 4)
    inside = intersection(boxes, boxes) / np.maximum(area(boxes), 1e-9)[:, None]
    keep = []
    for i in range(len(boxes)):
        if all(inside[i, k] < INSIDE for k in keep):
            keep.append(i)
    things = [things[i] for i in keep]
    # scene parts: may contain things (a chair in front of a wall), so they don't swallow them,
    # but a thing that *is* the window / door outline is dropped
    scene = windows(gray) + doors(raw_edges, gray) + wall(raw_edges, gray)
    if scene and things:
        sb = np.array([b for b, _ in scene], np.float32)
        same = iou(np.array([b for b, _ in things], np.float32), sb) >= 0.5
        things = [d for d, dup in zip(things, same.any(1)) if not dup]
    found = (scene + things)[:MAX_TRACKS * 2]
    return np.array([b for b, _ in found], np.float32).reshape(-1, 4), [lab for _, lab in found]


def compatible(track_label, det_label):
    """Walls, doors and windows only ever match their own kind, so a chair track can't slide onto
    the wall behind it; everything else may (the model sometimes flips chair <-> table)."""
    if track_label in SCENE or det_label in SCENE:
        return track_label == det_label
    return True


def warp_boxes(boxes, m, size):
    """Move boxes (N, 4 fractions of the picture) by a 2x3/3x3 pixel transform; returns new boxes."""
    if len(boxes) == 0:
        return boxes
    w, h = size
    b = np.asarray(boxes, np.float32) * (w, h, w, h)
    corners = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [0, 3]], b[:, [2, 3]]], 1)   # (N, 4, 2)
    moved = corners @ m[:2, :2].T + m[:2, 2]
    out = np.concatenate([moved.min(1), moved.max(1)], 1) / (w, h, w, h)
    return out.astype(np.float32)


class Motion:
    """How the whole picture moved between frames - mostly the drone itself moving and turning.
    One similarity transform per frame (shift + turn + zoom) from sparse optical flow, like
    drift.py. A short history lets a late model result be carried forward to the newest frame."""

    def __init__(self):
        self.prev = None
        self.history = deque(maxlen=MOTION_HISTORY)   # (frame time, transform since the first frame)
        self.ok = False                                # last frame's motion was measured

    def update(self, gray, t):
        """Transform (3x3, work-size pixels) from the previous frame to this one."""
        step = np.eye(3, dtype=np.float32)
        prev, self.prev = self.prev, gray
        self.ok = False
        if prev is not None and prev.shape == gray.shape:
            p0 = cv2.goodFeaturesToTrack(prev, maxCorners=120, qualityLevel=0.01, minDistance=8)
            if p0 is not None and len(p0) >= MIN_FLOW_POINTS:
                p1, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
                ok = ok.ravel() == 1
                if ok.sum() >= MIN_FLOW_POINTS:
                    m, inliers = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                                             ransacReprojThreshold=2.0)
                    if m is not None and inliers.sum() >= MIN_FLOW_POINTS:
                        step[:2] = m
                        self.ok = True
        else:
            self.history.clear()
        total = step @ self.history[-1][1] if self.history else np.eye(3, dtype=np.float32)
        self.history.append((t, total))
        return step

    def since(self, t):
        """Transform from the frame at time t to the newest frame, or None if it's too old."""
        for ft, total in self.history:
            if ft == t:
                return self.history[-1][1] @ np.linalg.inv(total)
        return None


class Track:
    def __init__(self, tid, box, t, label=None, score=0.0, model=False):
        self.id = tid
        self.votes = {}                          # name -> decayed sum of scores
        self.shown = None                        # the name on screen (sticky, see vote)
        self.model_hits = 0                      # times the model itself saw this track
        self.model_t = None                      # when it last did
        self.vote(label, score, model)
        self.box = np.array(box, np.float32)
        self.vel = np.zeros(2, np.float32)       # own motion on top of the camera's, picture fractions / s
        self.growth = 0.0                        # d ln(size) / dt, per s (+ = getting bigger)
        self.last_size = self.size
        self.hits = 1
        self.born = self.seen = self.updated = t

    def vote(self, label, score, model=False):
        if not label:
            return
        self.model_hits += model
        self.votes = {k: v * VOTE_DECAY for k, v in self.votes.items()}
        self.votes[label] = self.votes.get(label, 0.0) + score
        best = max(self.votes, key=self.votes.get)
        # hysteresis: the shown name only changes when another clearly beats it
        if self.shown is None or self.votes[best] > SWITCH * self.votes.get(self.shown, 0.0):
            self.shown = best

    @property
    def label(self):
        return self.shown or "object"

    @property
    def named(self):
        """Sure enough of its name to show it."""
        return self.shown is not None and self.votes.get(self.shown, 0.0) >= NAME_MIN

    @property
    def center(self):
        return (self.box[:2] + self.box[2:]) / 2

    @property
    def size(self):
        return math.sqrt(max(1e-9, float((self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))))

    @property
    def ttc(self):
        """Seconds until we reach it at the current closing speed, or None if not closing."""
        return 1.0 / self.growth if self.growth > 1.0 / TTC_SHOW else None

    @property
    def coast(self):
        return COAST_NAMED if self.model_hits >= 2 or self.named else COAST

    def predict(self, t, step, size):
        """Carry the box along with the camera's motion, plus its own."""
        dt = t - self.updated
        self.updated = t
        self.box = warp_boxes(self.box[None], step, size)[0]
        shift = self.vel * dt
        self.box += np.concatenate([shift, shift])

    def correct(self, box, t, weight):
        dt = max(1e-3, t - self.seen)
        box = np.asarray(box, np.float32)
        miss = ((box[:2] + box[2:]) / 2 - self.center) / dt     # where it is vs where we expected
        self.box += weight * (box - self.box)
        self.vel = np.clip(self.vel + VEL_GAIN * miss, -MAX_VEL, MAX_VEL)
        a_g = 1 - math.exp(-dt / GROWTH_TAU)
        self.growth += a_g * (math.log(self.size / self.last_size) / dt - self.growth)
        self.last_size = self.size
        self.hits += 1
        self.seen = t


class ObstacleTracker:
    def __init__(self, camera="forward", model="tiny", log=print):
        """model: "tiny" / "nano" (see detector.py), or None for unnamed boxes only."""
        self.camera = camera
        self.enabled = True
        self.tracks = []
        self.next_id = 1
        self.ms = 0.0            # time spent on the last frame (generic detector + tracking)
        self.namer = detector.AsyncDetector(model, log) if model else None
        self.motion = Motion()
        self.sharpness = None    # running average of how sharp frames are
        self.blurry = False
        self.named = np.zeros((0, 4), np.float32)    # latest model boxes (moved with the camera)
        self.named_t = -1e9

    @property
    def model_status(self):
        """(text, ok) for the HUD: which model is naming things and how fast."""
        if self.namer is None:
            return "NO NAMES", True
        if self.namer.status == "ready":
            return f"{self.namer.name.upper()} {self.namer.ms:.0f} ms", True
        return ("MODEL LOADING" if self.namer.status == "loading" else "NO MODEL"), False

    def update(self, frame, t):
        if not self.enabled:
            self.tracks = []
            self.motion = Motion()
            return
        t0 = cv2.getTickCount()
        small = shrink(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        step = self.motion.update(gray, t)

        # a frame smeared by a sudden move gives junk boxes: skip detecting on it, just follow
        sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        self.sharpness = sharp if self.sharpness is None else self.sharpness + 0.05 * (sharp - self.sharpness)
        self.blurry = sharp < BLUR_DROP * self.sharpness

        for tr in self.tracks:
            tr.predict(t, step, size)
        self.named = warp_boxes(self.named, step, size)

        dets, labels = np.zeros((0, 4), np.float32), []      # labels: (name, score, from the model)
        if self.namer:
            if not self.blurry:
                self.namer.submit(frame, t)
            res = self.namer.poll()
            if res is not None:
                named, ft = res
                m = self.motion.since(ft)                    # it was seen a few frames ago
                if m is not None:
                    dets = warp_boxes(np.array([b for b, _, _ in named], np.float32).reshape(-1, 4), m, size)
                    labels = [(lab, sc, True) for _, lab, sc in named]
                    self.named, self.named_t = dets, t
        if not self.blurry:
            shapes, kinds = _detect(small)
            if t - self.named_t <= NAMED_FRESH and len(self.named) and len(shapes):
                # shapes on top of a named thing are pieces of it (walls / doors / windows may hold one)
                inside = intersection(shapes, self.named) / np.maximum(area(shapes), 1e-9)[:, None]
                scene = np.array([k in SCENE for k in kinds], bool)
                ok = ((inside < INSIDE).all(1) & (iou(shapes, self.named) < MATCH_IOU).all(1)) | scene
                shapes, kinds = shapes[ok], [k for k, o in zip(kinds, ok) if o]
            dets = np.concatenate([dets, shapes])
            labels += [(k, HEURISTIC_VOTE if k else 0.0, False) for k in kinds]

        unmatched = set(range(len(dets)))
        if self.tracks and len(dets):
            boxes = np.array([tr.box for tr in self.tracks])
            overlap = iou(boxes, dets)
            centres = (boxes[:, None, :2] + boxes[:, None, 2:]) / 2
            dist = np.linalg.norm(centres - (dets[None, :, :2] + dets[None, :, 2:]) / 2, axis=2)
            score = np.where(overlap >= MATCH_IOU, 1 + overlap, np.where(dist <= MATCH_DIST, 1 - dist, 0))
            score *= [[compatible(tr.label if tr.votes else None, lab) and
                       not (model and lab and tr.named and lab != tr.label)   # chair in front of a table
                       for lab, _, model in labels] for tr in self.tracks]
            # greedy: model boxes pick first, then best overlap. Older tracks win ties (stable IDs).
            for ti, di in sorted(zip(*np.nonzero(score)), key=lambda p: (not labels[p[1]][2], -score[p], p[0])):
                tr = self.tracks[ti]
                if di in unmatched and tr.seen != t:
                    lab, sc, model = labels[di]
                    tr.correct(dets[di], t, MODEL_WEIGHT if model else SHAPE_WEIGHT)
                    if model:
                        tr.model_t = t
                    tr.vote(lab, sc, model)
                    unmatched.discard(di)
        self.tracks = [tr for tr in self.tracks if t - tr.seen <= tr.coast and on_screen(tr.box)]
        held = np.array([tr.box for tr in self.tracks]).reshape(-1, 4)
        for di in sorted(unmatched):
            lab, sc, model = labels[di]
            if model and sc < BIRTH_CONF:                    # weak model boxes only keep tracks going
                continue
            if len(held) and iou(dets[di:di + 1], held).max() >= BIRTH_BLOCK:
                continue                                     # a second box on something already tracked
            if len(self.tracks) >= MAX_TRACKS:
                break
            self.tracks.append(Track(self.next_id, dets[di], t, lab, sc, model))
            self.next_id += 1
        self._merge()
        self.ms = (cv2.getTickCount() - t0) * 1000 / cv2.getTickFrequency()

    def _merge(self):
        """Two tracks sitting on the same thing become one: the older keeps its ID."""
        if len(self.tracks) < 2:
            return
        self.tracks.sort(key=lambda tr: tr.born)
        boxes = np.array([tr.box for tr in self.tracks])
        same = iou(boxes, boxes) >= MERGE_IOU
        gone = set()
        for i, keep in enumerate(self.tracks):
            for j in range(i + 1, len(self.tracks)):
                other = self.tracks[j]
                if i in gone or j in gone or not same[i, j]:
                    continue
                if not compatible(keep.label if keep.votes else None, other.label if other.votes else None):
                    continue
                if keep.model_t is not None and keep.model_t == other.model_t:
                    continue                 # the model saw two separate things (chairs side by side)
                for lab, v in other.votes.items():
                    keep.votes[lab] = keep.votes.get(lab, 0.0) + v
                if keep.shown is None:
                    keep.shown = other.shown
                keep.model_hits += other.model_hits
                keep.hits = max(keep.hits, other.hits)
                keep.seen = max(keep.seen, other.seen)
                gone.add(j)
        self.tracks = [tr for k, tr in enumerate(self.tracks) if k not in gone]

    def is_threat(self, tr):
        if self.camera != "forward" or tr.label == "door":     # a door is the way through
            return False
        x0, _, x1, _ = tr.box
        in_path = x1 > 0.5 - CORRIDOR / 2 and x0 < 0.5 + CORRIDOR / 2
        near = tr.box[3] - tr.box[1] >= NEAR
        return bool(in_path and (near or (tr.ttc is not None and tr.ttc < TTC_WARN)))

    def objects(self):
        """Confirmed tracks for the HUD: dicts with id, label, box (x0 y0 x1 y1 fractions), ttc, threat.
        Unnamed (or not yet surely named) things are only listed while they're in the way, as
        "obstacle"."""
        out = []
        for tr in self.tracks:
            threat = self.is_threat(tr)
            if tr.hits >= CONFIRM and (tr.named or threat):
                out.append({"id": tr.id, "label": tr.label if tr.named else "obstacle",
                            "box": tuple(float(v) for v in tr.box), "ttc": tr.ttc, "threat": threat,
                            "lost": tr.seen != tr.updated})
        return out


def on_screen(box):
    return box[2] > 0 and box[3] > 0 and box[0] < 1 and box[1] < 1


if __name__ == "__main__":
    # Try it on a webcam or a video file: python obstacles.py [index-or-path] [tiny|nano|off]
    import sys
    import time
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    model = sys.argv[2] if len(sys.argv) > 2 else "tiny"
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    tracker = ObstacleTracker(model=None if model == "off" else model)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracker.update(frame, time.time())
        h, w = frame.shape[:2]
        for o in tracker.objects():
            x0, y0, x1, y1 = o["box"]
            color = (70, 70, 240) if o["threat"] else (90, 220, 90)
            cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), color, 2)
            label = f"{o['label']} #{o['id']}" + (f" {o['ttc']:.1f}s" if o["ttc"] else "")
            cv2.putText(frame, label, (int(x0 * w), int(y0 * h) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"{tracker.ms:.1f} ms  {tracker.model_status[0]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        cv2.imshow("obstacles", frame)
        if cv2.waitKey(1) == 27:
            break
