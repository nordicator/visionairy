"""Metric depth from the drone's single camera: how far away every pixel is, in metres.

Model: Depth Anything V2 Metric Indoor, Small (trained on indoor scenes, 0-20 m), exported to ONNX
by export_depth.py into models/. Runs with onnxruntime on the GPU (DirectML: AMD / Intel /
NVIDIA on Windows) when there is one, else on the CPU, on a background thread so the video never
waits. A single camera can't measure distance the way two eyes can; the model judges it from how
things look. Four steps make its metres trustworthy:

  1. Lens correction   The model assumes the camera it was trained with (60 deg wide). A wider
                       lens makes everything look smaller, so farther. Depth is multiplied by
                       (our focal length / training focal length), both relative to image width.
                       Focal length comes from camera.json (calibrate_camera.py, exact) or a
                       field-of-view estimate (CAMERA_HFOV, rough).
  2. Tape calibration  Point the camera square at a wall a known distance away and press M in
                       drone.py: the remaining scale error of this camera is measured and saved
                       to depth_cal.json. Several presses at different distances are averaged.
  3. Steadiness        Each new depth map is moved with the camera's motion since the last one
                       (from obstacles.Motion) and blended in where they agree, so single-frame
                       flicker averages out while real changes come through.
  4. Robust sampling   An object's distance is a low percentile of the depth inside the middle
                       of its box - the near surface of the object, not the wall behind it.

Coordinates (camera frame, metres): x right, y down, z forward (= distance along the view).
"""
import json
import math
import os
import threading
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
CAMERA_FILE = os.path.join(HERE, "camera.json")
CAL_FILE = os.path.join(HERE, "depth_cal.json")
# (height, width) per quality, one per picture shape; the one closest to the camera's shape is used
# so the picture isn't squashed (that skews distances). GPU (Radeon 860M): fast ~55 ms, fine ~430 ms.
MODELS = {"fast": [(252, 420), (294, 392)],     # 5:3 drone camera, 4:3 webcam
          "fine": [(378, 630), (378, 504)]}
# fallback if export_depth.py was never run: a community ONNX export of the same official weights
# (Apache-2.0), fixed square 518x518 input - slower, and squashes the picture a little
FALLBACK_URL = "https://huggingface.co/77ukhtar/depth-anything-v2-metric-onnx/resolve/main/model.onnx"
FALLBACK_NAME = "depth_indoor_s_518x518.onnx"

TRAIN_FOCAL = 886.81 / 1024    # Hypersim (the training set): 60 deg horizontal, focal / width
CAMERA_HFOV = 90.0             # deg, a guess for toy-drone cameras until camera.json exists
MAX_DEPTH = 20.0               # m, the model's range
BLEND = 0.45                   # how much a new depth map moves the running one
AGREE = 0.25                   # relative difference under which new and old depth "agree"
OBJECT_PCT = 25                # percentile of depth inside a box = its near surface
OBJECT_INNER = 0.6             # use the middle 60% of a box (edges catch the background)
VIEW_NEAR, VIEW_FAR = 0.3, 5.0 # m, colour range of the depth view (red .. blue, log scale)


class Camera:
    """Focal length and centre, as fractions of the picture width / height, plus lens distortion."""

    def __init__(self, hfov=None, aspect=640 / 384):
        hfov = hfov or CAMERA_HFOV
        self.fx = 0.5 / math.tan(math.radians(hfov) / 2)     # focal length / picture width
        self.fy = self.fx * aspect                            # focal length / picture height
        self.cx = self.cy = 0.5
        self.dist = None
        self.source = f"guess ({hfov:.0f} deg wide)"
        self.guessed = True
        try:
            with open(CAMERA_FILE) as f:
                c = json.load(f)
            self.fx, self.fy, self.cx, self.cy = c["fx"], c["fy"], c["cx"], c["cy"]
            self.dist = np.array(c.get("dist", []), np.float64) if c.get("dist") else None
            self.source = "calibrated"
            self.guessed = False
        except (OSError, ValueError, KeyError):
            pass
        self._maps = None

    @property
    def hfov(self):
        return math.degrees(2 * math.atan(0.5 / self.fx))

    def undistort(self, frame):
        """Straighten wide-angle bending (only once calibrate_camera.py has measured it)."""
        if self.dist is None or not self.dist.any():
            return frame
        h, w = frame.shape[:2]
        if self._maps is None or self._maps[0].shape[:2] != (h, w):
            k = np.array([[self.fx * w, 0, self.cx * w], [0, self.fy * h, self.cy * h], [0, 0, 1]])
            self._maps = cv2.initUndistortRectifyMap(k, self.dist, None, k, (w, h), cv2.CV_16SC2)
        return cv2.remap(frame, *self._maps, cv2.INTER_LINEAR)

    def point(self, u, v, z):
        """(u, v) fractions of the picture at distance z -> (x, y, z) metres."""
        return (u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z


def load_cal():
    try:
        with open(CAL_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"scale": 1.0, "samples": []}


def save_cal(cal):
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=1)


def model_path(quality, aspect=640 / 384, log=print):
    """The exported model whose shape is nearest the picture's (width / height)."""
    for q in (quality, "fast" if quality == "fine" else "fine"):
        sizes = sorted(MODELS[q], key=lambda hw: abs(math.log(hw[1] / hw[0] / aspect)))
        for h, w in sizes:
            path = os.path.join(MODEL_DIR, f"depth_indoor_s_{h}x{w}.onnx")
            if os.path.exists(path):
                return path
    fallback = os.path.join(MODEL_DIR, FALLBACK_NAME)
    if not os.path.exists(fallback):
        log("depth model not exported (python export_depth.py); downloading a ready-made one (99 MB) ...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(FALLBACK_URL, fallback + ".part")
        os.replace(fallback + ".part", fallback)
    return fallback


class DepthModel:
    def __init__(self, quality="fast", camera=None, log=print, aspect=640 / 384):
        import onnxruntime as ort
        self.camera = camera or Camera()
        path = model_path(quality, aspect, log)
        providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(path, opts, providers=providers)
        self.device = "GPU" if self.session.get_providers()[0] != "CPUExecutionProvider" else "CPU"
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.h, self.w = inp.shape[2], inp.shape[3]
        self.normalised = "518x518" in path          # the fallback export wants ImageNet-normalised input
        self.cal = load_cal()

    @property
    def lens_factor(self):
        """Multiply the model's metres by this to account for our lens being wider / narrower."""
        return self.camera.fx / TRAIN_FOCAL

    def raw(self, frame):
        """Model metres for the frame, before lens and tape calibration, at model size (h, w)."""
        img = cv2.resize(self.camera.undistort(frame), (self.w, self.h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.normalised:
            rgb = (rgb / 255 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], np.float32)
        return self.session.run(None, {self.input_name: x})[0][0].astype(np.float32)

    def predict(self, frame):
        """Metres, at model size."""
        return self.raw(frame) * (self.lens_factor * self.cal.get("scale", 1.0))

    def calibrate(self, frame, true_distance):
        """The camera faces a flat wall true_distance metres away: learn this camera's scale.
        Returns the new scale; samples are kept, and the median of them is used."""
        d = self.raw(frame) * self.lens_factor
        h, w = d.shape
        measured = float(np.median(d[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]))
        samples = self.cal.get("samples", [])[-9:] + [[true_distance, measured]]
        self.cal = {"scale": float(np.median([t / m for t, m in samples])), "samples": samples,
                    "lens": self.camera.source}
        save_cal(self.cal)
        return self.cal["scale"]


class DepthMap:
    """The running (blended) depth map: metres, at model size, for the frame taken at time t."""

    def __init__(self, depth, t, camera):
        self.depth, self.t, self.camera = depth, t, camera

    def box(self, box, pct=OBJECT_PCT):
        """Distance (m) to the near surface of the thing in box (x0 y0 x1 y1 fractions), or None."""
        h, w = self.depth.shape
        x0, y0, x1, y1 = box
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * OBJECT_INNER, (y1 - y0) * OBJECT_INNER
        c0, c1 = int(max(0, cx - bw / 2) * w), int(min(1, cx + bw / 2) * w)
        r0, r1 = int(max(0, cy - bh / 2) * h), int(min(1, cy + bh / 2) * h)
        if c1 - c0 < 2 or r1 - r0 < 2:
            return None
        return float(np.percentile(self.depth[r0:r1, c0:c1], pct))

    def ahead(self, corridor=0.3, pct=10):
        """Nearest thing (m) in the middle of the view - what the drone would hit flying straight."""
        h, w = self.depth.shape
        region = self.depth[int(h * 0.3):int(h * 0.7), int(w * (0.5 - corridor / 2)):int(w * (0.5 + corridor / 2))]
        return float(np.percentile(region, pct))

    def at(self, u, v):
        """Distance (m) at one point (fractions of the picture), from a small median patch."""
        h, w = self.depth.shape
        c, r = int(min(max(u, 0), 0.999) * w), int(min(max(v, 0), 0.999) * h)
        return float(np.median(self.depth[max(0, r - 2):r + 3, max(0, c - 2):c + 3]))

    def colour(self, size, near=None, far=None):
        """Picture of the depth map (near = red, far = blue) for the HUD."""
        near, far = near or VIEW_NEAR, far or VIEW_FAR
        d = np.clip((np.log(self.depth) - math.log(near)) / (math.log(far) - math.log(near)), 0, 1)
        img = cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


class AsyncDepth:
    """DepthModel on a background thread, always on the newest frame, with temporal blending.

    submit(frame, t) frames as they come; poll(motion) returns a new DepthMap when one is ready.
    motion (obstacles.Motion) moves the previous map onto the new frame before blending."""

    def __init__(self, quality="fast", log=print):
        self.quality, self.log = quality, log
        self.status = "loading"     # loading | ready | failed
        self.device = ""
        self.ms = 0.0
        self.model = None
        self.camera = Camera()
        self.current = None         # latest blended DepthMap
        self._frame = None
        self._result = None
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame, t):
        if self.camera.guessed:                  # a guessed lens: fit it to the picture's shape
            self.camera.fy = self.camera.fx * frame.shape[1] / frame.shape[0]
        with self._cv:
            self._frame = (frame, t)
            self._cv.notify()

    def poll(self, motion=None, size=None):
        """Blend in a finished depth map, if any. motion: obstacles.Motion; size: its pixel size."""
        with self._cv:
            res, self._result = self._result, None
        if res is None:
            return None
        depth, t = res
        prev = self.current
        if prev is not None and motion is not None and size is not None:
            m = _between(motion, prev.t, t)
            if m is not None:
                h, w = depth.shape
                sx, sy = w / size[0], h / size[1]
                s = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float64)
                mm = (s @ m @ np.linalg.inv(s))[:2]
                zoom = math.hypot(mm[0, 0], mm[1, 0])          # flying forward: things got nearer
                old = cv2.warpAffine(prev.depth, mm, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) / max(zoom, 1e-3)
                agree = (old > 0) & (np.abs(depth - old) < AGREE * depth)
                depth = np.where(agree, old + BLEND * (depth - old), depth).astype(np.float32)
        self.current = DepthMap(depth, t, self.camera)
        return self.current

    def calibrate(self, frame, true_distance):
        if self.model is None:
            return None
        return self.model.calibrate(frame, true_distance)

    def _run(self):
        with self._cv:                  # wait for a frame: the model is picked to match its shape
            while self._frame is None:
                self._cv.wait()
            h, w = self._frame[0].shape[:2]
        try:
            self.model = DepthModel(self.quality, self.camera, self.log, w / h)
        except Exception as e:      # no onnxruntime, no model, no internet ...
            self.status = "failed"
            self.log(f"depth unavailable ({e})")
            return
        self.device = self.model.device
        self.status = "ready"
        self.log(f"depth: {self.model.w}x{self.model.h} on {self.device}, lens {self.camera.source} "
                 f"({self.camera.hfov:.0f} deg), scale x{self.model.cal.get('scale', 1.0):.2f}")
        while True:
            with self._cv:
                while self._frame is None:
                    self._cv.wait()
                (frame, t), self._frame = self._frame, None
            t0 = time.perf_counter()
            depth = np.clip(self.model.predict(frame), 0.05, MAX_DEPTH * 2)
            self.ms = (time.perf_counter() - t0) * 1000
            with self._cv:
                self._result = (depth, t)


def _between(motion, t_from, t_to):
    """Camera motion (3x3, motion's pixels) from the frame at t_from to the one at t_to, or None."""
    a = b = None
    for ft, total in motion.history:
        if ft == t_from:
            a = total
        if ft == t_to:
            b = total
    if a is None or b is None:
        return None
    return b @ np.linalg.inv(a)
