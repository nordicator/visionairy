"""Measure the drone camera's lens (focal length, centre, distortion) from checkerboard photos.

depth.py needs the focal length to turn the depth model's metres into this camera's metres, and
to place objects left/right/up/down. Without camera.json it guesses a 90 deg wide lens, which can
be 20-40% off. This makes it exact.

  1. Print a checkerboard (e.g. "opencv chessboard 9x6" - 10 x 7 squares, so 9 x 6 inner
     corners). Tape it flat to a board.
  2. Save frames while showing it to the drone camera from 15-30 different angles and distances,
     filling different parts of the picture:   python drone.py probe --save calib/
  3. python calibrate_camera.py calib/            (or --cols 9 --rows 6 for another board)
     -> writes camera.json next to this script. drone.py picks it up on the next start.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

import depth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder of saved frames (JPEG / PNG)")
    ap.add_argument("--cols", type=int, default=9, help="inner corners across (default 9)")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down (default 6)")
    ap.add_argument("--webcam", action="store_true", help="these are webcam frames: write camera_webcam.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.jpg")) + glob.glob(os.path.join(args.folder, "*.png")))
    pattern = (args.cols, args.rows)
    grid = np.zeros((args.cols * args.rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    obj, img, size, last = [], [], None, None
    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        if last is not None and np.abs(corners - last).mean() < 4:
            continue                                  # nearly the same view as the last one
        last = corners
        obj.append(grid)
        img.append(corners)
    print(f"checkerboard found in {len(img)} of {len(files)} frames")
    if len(img) < 8:
        raise SystemExit("need at least 8 different views of the board (15-30 is better)")
    rms, k, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    w, h = size
    cam = {"fx": k[0, 0] / w, "fy": k[1, 1] / h, "cx": k[0, 2] / w, "cy": k[1, 2] / h,
           "dist": dist.ravel().tolist(), "size": [w, h], "rms_px": rms, "views": len(img)}
    if args.webcam:
        depth.CAMERA_FILE = os.path.join(os.path.dirname(depth.CAMERA_FILE), "camera_webcam.json")
    with open(depth.CAMERA_FILE, "w") as f:
        json.dump(cam, f, indent=1)
    hfov = np.degrees(2 * np.arctan(0.5 / cam["fx"]))
    print(f"reprojection error {rms:.2f} px ({'good' if rms < 0.7 else 'ok' if rms < 1.5 else 'poor - retake'}), "
          f"{hfov:.1f} deg wide; saved {depth.CAMERA_FILE}")
    print("Old depth calibrations (depth_cal.json) assumed the old lens - redo key M once.")


if __name__ == "__main__":
    main()
