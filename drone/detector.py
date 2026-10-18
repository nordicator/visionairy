"""Named object detection (human, chair, table, tv, bag) with a small YOLOX model run by OpenCV's DNN.

The model is trained on COCO's 80 everyday classes; only the ones in KEEP are reported, under the
names given there. (COCO has no box, window, door or wall - obstacles.py finds those by shape.)
It's an ONNX file that's downloaded once into models/ on first use:

  tiny   20 MB  ~30 ms per frame on a laptop CPU   (default, better at small / far things)
  nano    4 MB  ~15 ms per frame                   (for slow machines)

AsyncDetector runs the model on a background thread, always on the newest frame it was handed,
so a slow CPU makes labels update less often but never stalls the video window.
"""
import os
import threading
import time
import urllib.request

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_{}.onnx"
MODELS = ("tiny", "nano")
INPUT = 416               # model input size (square, letterboxed)
STRIDES = (8, 16, 32)
CONF = 0.25               # min objectness x class score (the tracker needs more to start a track)
NMS_IOU = 0.45
THREADS = 4               # CPU threads for OpenCV; leaves the rest for video + UI

COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush")
# COCO class -> name shown. Everything else the model sees is ignored.
KEEP = {"person": "human", "chair": "chair", "table": "table", "tv": "tv",
        "backpack": "bag", "handbag": "bag", "suitcase": "bag"}


def model_path(name):
    """Local path of the model, downloading it the first time."""
    path = os.path.join(MODEL_DIR, f"yolox_{name}.onnx")
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URL.format(name), tmp)
        os.replace(tmp, path)
    return path


def _grids():
    grids, strides = [], []
    for s in STRIDES:
        n = INPUT // s
        xs, ys = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


class Detector:
    def __init__(self, name="tiny"):
        self.net = cv2.dnn.readNetFromONNX(model_path(name))
        cv2.setNumThreads(THREADS)
        self.grid, self.stride = _grids()

    def detect(self, frame):
        """[(box x0 y0 x1 y1 as fractions of the picture, label, score), ...]"""
        h, w = frame.shape[:2]
        r = min(INPUT / h, INPUT / w)
        nw, nh = round(w * r), round(h * r)
        padded = np.full((INPUT, INPUT, 3), 114, np.uint8)       # letterbox, image top-left
        padded[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.net.setInput(cv2.dnn.blobFromImage(padded))         # BGR, 0-255, as YOLOX expects
        out = self.net.forward()[0]
        cls = out[:, 5:].argmax(1)
        score = out[:, 4] * out[np.arange(len(out)), 5 + cls]
        keep = score >= CONF
        if not keep.any():
            return []
        out, cls, score = out[keep], cls[keep], score[keep]
        xy = (out[:, :2] + self.grid[keep]) * self.stride[keep] / r
        wh = np.exp(out[:, 2:4]) * self.stride[keep] / r
        xywh = np.concatenate([xy - wh / 2, wh], 1)
        # across classes, so one thing never gets two labels (the best one wins)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), CONF, NMS_IOU)
        dets = []
        for i in np.array(idx, int).ravel():
            name = KEEP.get(COCO[cls[i]])
            if name is None:
                continue
            x, y, bw, bh = xywh[i]
            box = (max(0.0, x / w), max(0.0, y / h), min(1.0, (x + bw) / w), min(1.0, (y + bh) / h))
            dets.append((box, name, float(score[i])))
        return dets


class AsyncDetector:
    """Detector on a background thread. submit() frames as they come, poll() for new results."""

    def __init__(self, name="tiny", log=print):
        self.name, self.log = name, log
        self.status = "loading"     # loading | ready | failed
        self.ms = 0.0               # time per frame of the model
        self._frame = None
        self._result = None         # (dets, frame time), until poll() takes it
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame, t):
        with self._cv:
            self._frame = (frame, t)
            self._cv.notify()

    def poll(self):
        """(dets, time of the frame they came from) if a new result is ready, else None."""
        with self._cv:
            res, self._result = self._result, None
            return res

    def _run(self):
        try:
            if not os.path.exists(os.path.join(MODEL_DIR, f"yolox_{self.name}.onnx")):
                self.log(f"downloading the {self.name} object model (once) ...")
            det = Detector(self.name)
        except Exception as e:      # no internet, bad file, ... -> unlabelled tracking only
            self.status = "failed"
            self.log(f"object model unavailable ({e}); boxes will be unlabelled")
            return
        self.status = "ready"
        while True:
            with self._cv:
                while self._frame is None:
                    self._cv.wait()
                (frame, t), self._frame = self._frame, None
            t0 = time.perf_counter()
            dets = det.detect(frame)
            self.ms = (time.perf_counter() - t0) * 1000
            with self._cv:
                self._result = (dets, t)
