"""Heads-up display for the drone video window, drawn with OpenCV.

draw(canvas, state) paints everything on top of the video; drone.py builds `state` each frame.
"""
import cv2
import numpy as np

W, H = 960, 540
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
DARK = (25, 25, 25)
CAP = (60, 60, 60)
ACCENT = (0, 200, 255)       # amber (BGR)
GREEN = (90, 220, 90)
YELLOW = (0, 220, 255)
RED = (70, 70, 240)

ACTION_STYLE = {1: ("TAKING OFF", GREEN), 2: ("LANDING", YELLOW), 4: ("MOTORS OFF", RED),
                0x80: ("CALIBRATING GYRO", YELLOW)}
AUTO_STYLE = {"learning": ("CORRECTING", YELLOW), "steady": ("HOLDING", GREEN), "paused": ("PAUSED", GREY),
              "waiting": ("AFTER TAKEOFF", GREY), "low texture": ("LOW TEXTURE", RED), "off": ("AUTO OFF", GREY)}
DRIFT_FULL = 0.15         # frame-widths/s drawn as a full-length drift arrow
PANEL_W = 180             # width of the right-hand panels


def fit(frame):
    """Letterbox a video frame (or nothing) onto the W x H canvas."""
    canvas = np.zeros((H, W, 3), np.uint8)
    if frame is None:
        return canvas
    h, w = frame.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    x, y = (W - nw) // 2, (H - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def panel(img, x0, y0, x1, y1, alpha=0.6):
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def text(img, s, org, color=WHITE, scale=0.5, thick=1, anchor="l"):
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if anchor == "c":
        x -= tw // 2
    elif anchor == "r":
        x -= tw
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return tw


def boxed(img, s, center, color, scale=0.8, thick=2, pad=10):
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
    cx, cy = center
    panel(img, cx - tw // 2 - pad, cy - th - pad, cx + tw // 2 + pad, cy + base + pad // 2, 0.7)
    text(img, s, (cx, cy), color, scale, thick, "c")


def keycap(img, x, y, w, h, label, down, caption=None, arrow=None):
    fill, edge, ink = (ACCENT, WHITE, DARK) if down else (CAP, (110, 110, 110), WHITE)
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), edge, 1, cv2.LINE_AA)
    cx, cy = x + w // 2, y + h // 2
    if arrow:
        d = -1 if arrow == "up" else 1
        pts = np.array([(cx, cy + 8 * d), (cx - 8, cy - 6 * d), (cx + 8, cy - 6 * d)], np.int32)
        cv2.fillPoly(img, [pts], ink, cv2.LINE_AA)
        return
    if caption:
        lw = text(img, label, (x + 7, cy + 5), ink, 0.45, 1)
        text(img, caption, (x + 14 + lw, cy + 5), ink, 0.42, 1)
    else:
        text(img, label, (cx, cy + 6), ink, 0.55, 1, "c")


def crosshair(img):
    cx, cy = W // 2, H // 2
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (cx + 6 * dx, cy + 6 * dy), (cx + 16 * dx, cy + 16 * dy), (200, 200, 200), 1, cv2.LINE_AA)


def signal_bars(img, x, y, pct):
    for i in range(4):
        h = 4 + i * 3
        on = pct is not None and pct > i * 25
        cv2.rectangle(img, (x + i * 5, y - h), (x + i * 5 + 3, y), GREEN if on else (90, 90, 90), -1)


def top_bar(img, st):
    panel(img, 0, 0, W, 30)
    label, color = st["status"]
    blink = label != "SIGNAL LOST" or int(st["t"] * 3) % 2
    if blink:
        cv2.circle(img, (16, 15), 6, color, -1, cv2.LINE_AA)
    x = 30 + text(img, label, (30, 20), color, 0.5, 1)
    x += 18 + text(img, f"VARIANT {st['variant']}", (x + 18, 20), GREY, 0.45)
    if st["wifi"]:
        ssid, sig = st["wifi"]
        signal_bars(img, x + 18, 21, sig)
        text(img, f"{ssid}  {sig}%" if sig is not None else ssid, (x + 42, 20), GREY, 0.45)
    if not st["fly"]:
        text(img, "PROBE MODE - no flight commands", (W // 2, 20), YELLOW, 0.45, 1, "c")

    right = W - 12
    items = []
    if st["age_ms"] is not None:
        a = st["age_ms"]
        items.append((f"{a} ms", GREEN if a < 300 else YELLOW if a < 1000 else RED))
    if st["res"]:
        items.append((st["res"], GREY))
    items.append((f"{st['fps']:.0f} fps", GREY))
    if st["flight"] is not None:
        m, s = divmod(int(st["flight"]), 60)
        items.append((f"FLIGHT {m:02d}:{s:02d}", WHITE))
    if st["rec"] is not None:
        m, s = divmod(int(st["rec"]), 60)
        items.append((f"REC {m:02d}:{s:02d}", RED))
    for s, c in items:
        right -= text(img, s, (right, 20), c, 0.45, 1, "r") + 18
        if s.startswith("REC") and int(st["t"] * 2) % 2:
            cv2.circle(img, (right + 6, 15), 5, RED, -1, cv2.LINE_AA)
            right -= 14


def keys_panel(img, st):
    p = st["pressed"]
    if not st["show_keys"]:
        panel(img, 12, H - 40, 170, H - 12)
        text(img, "H  show controls", (22, H - 21), GREY, 0.45)
        return
    k, g = 38, 5
    rows_h = (2 * k + g + 24 + 36 + 32) if st["fly"] else 0
    x0, y0 = 12, H - 12 - (rows_h + 50)
    panel(img, x0, y0, x0 + 356, H - 12)
    x, y = x0 + 12, y0 + 12
    if st["fly"]:
        keycap(img, x + k + g, y, k, k, "W", "w" in p)
        for i, c in enumerate("asd"):
            keycap(img, x + i * (k + g), y + k + g, k, k, c.upper(), c in p)
        ax = x + 4 * (k + g)
        keycap(img, ax, y, k, k, "", p & {"up", "r"}, arrow="up")
        keycap(img, ax, y + k + g, k, k, "", p & {"down", "f"}, arrow="down")
        text(img, "up", (ax + k + 8, y + 24), GREY, 0.42)
        text(img, "down", (ax + k + 8, y + k + g + 24), GREY, 0.42)
        cy = y + 2 * k + g + 18
        text(img, "MOVE", (x + (3 * k + 2 * g) // 2, cy), GREY, 0.38, 1, "c")
        text(img, "HEIGHT (or R/F)", (ax + k // 2 - 18, cy), GREY, 0.38)
        y = cy + 8
        cx = x
        for key, name, cap, w in (("t", "T", "TAKE OFF", 104), ("l", "L", "LAND", 72),
                                  ("space", "SPACE", "OFF", 112)):
            keycap(img, cx, y, w, 28, name, key in p, cap)
            cx += w + 6
        y += 36
        cx = x
        for key, name, cap, w in (("c", "C", "CALIB", 70), ("k", "K", "AUTO-TRIM", 104),
                                  ("shift", "SHIFT", "+KEY TRIM", 124)):
            keycap(img, cx, y, w, 26, name, key in p, cap)
            cx += w + 6
        y += 32
    cx = x
    for key, name, cap, w in (("p", "P", "PHOTO", 72), ("v", "V", "REC", 58),
                              ("h", "H", "HIDE", 64), ("back", "BKSP", "TRIM 0", 90)):
        keycap(img, cx, y, w, 26, name, key in p, cap)
        cx += w + 6


def stick_panel(img, levels, trim):
    box, bar = 120, 22
    pw, ph = PANEL_W, 12 + 16 + box + 28
    x0, y0 = W - 12 - pw, H - 12 - ph
    panel(img, x0, y0, x0 + pw, y0 + ph)
    text(img, "X / Y", (x0 + 12, y0 + 22), GREY, 0.42)
    text(img, "Z", (x0 + 12 + box + 14 + bar // 2, y0 + 22), GREY, 0.42, 1, "c")
    bx, by = x0 + 12, y0 + 30
    cv2.rectangle(img, (bx, by), (bx + box, by + box), (110, 110, 110), 1)
    mid = box // 2
    cv2.line(img, (bx + mid, by), (bx + mid, by + box), (70, 70, 70), 1)
    cv2.line(img, (bx, by + mid), (bx + box, by + mid), (70, 70, 70), 1)
    dx = bx + mid + round(levels["x"] / 100 * (mid - 6))
    dy = by + mid - round(levels["y"] / 100 * (mid - 6))
    tx = bx + mid + round(trim["x"] / 100 * (mid - 6))
    ty = by + mid - round(trim["y"] / 100 * (mid - 6))
    cv2.circle(img, (tx, ty), 5, (255, 200, 120), 1, cv2.LINE_AA)       # trim offset
    cv2.line(img, (bx + mid, by + mid), (dx, dy), ACCENT, 2, cv2.LINE_AA)
    cv2.circle(img, (dx, dy), 6, ACCENT, -1, cv2.LINE_AA)
    zx = bx + box + 14
    cv2.rectangle(img, (zx, by), (zx + bar, by + box), (110, 110, 110), 1)
    cv2.line(img, (zx, by + mid), (zx + bar, by + mid), (70, 70, 70), 1)
    zend = by + mid - round(levels["z"] / 100 * (mid - 1))
    if zend != by + mid:
        cv2.rectangle(img, (zx + 2, min(zend, by + mid)), (zx + bar - 2, max(zend, by + mid)), ACCENT, -1)
    zt = by + mid - round(trim["z"] / 100 * (mid - 1))
    cv2.line(img, (zx - 4, zt), (zx + bar + 4, zt), (255, 200, 120), 1)
    text(img, f"X{levels['x']:+4.0f}  Y{levels['y']:+4.0f}  Z{levels['z']:+4.0f}",
         (x0 + pw // 2, by + box + 20), WHITE, 0.42, 1, "c")


def drift_panel(img, st):
    """Which way the drone slides on its own, and what auto-trim is doing about it."""
    ph = 150
    x0, y1 = W - 12 - PANEL_W, H - 12 - 176 - 8
    y0 = y1 - ph
    panel(img, x0, y0, x0 + PANEL_W, y1)
    text(img, "DRIFT", (x0 + 12, y0 + 20), GREY, 0.42)
    label, color = AUTO_STYLE.get(st["auto"], (st["auto"].upper(), GREY))
    text(img, label, (x0 + PANEL_W - 10, y0 + 20), color, 0.42, 1, "r")
    v, ok = st["drift"], st["drift_valid"]
    cx, cy, r = x0 + 50, y0 + 70, 34
    cv2.circle(img, (cx, cy), r, (110, 110, 110), 1, cv2.LINE_AA)
    text(img, "F", (cx, y0 + 33), (110, 110, 110), 0.35, 1, "c")
    ink = ACCENT if ok else (90, 90, 90)
    ex = cx + round(max(-1, min(1, v["x"] / DRIFT_FULL)) * r)
    ey = cy - round(max(-1, min(1, v["y"] / DRIFT_FULL)) * r)
    if (ex, ey) != (cx, cy):
        cv2.arrowedLine(img, (cx, cy), (ex, ey), ink, 2, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(img, (cx, cy), 3, ink, -1, cv2.LINE_AA)
    zx = x0 + 112
    cv2.line(img, (zx, cy - r), (zx, cy + r), (110, 110, 110), 1)
    text(img, "Z", (zx, y0 + 33), (110, 110, 110), 0.35, 1, "c")
    ez = cy - round(max(-1, min(1, v["z"] / DRIFT_FULL)) * r)
    if ez != cy:
        cv2.arrowedLine(img, (zx, cy), (zx, ez), ink, 2, cv2.LINE_AA, tipLength=0.3)
    text(img, "sliding:", (zx + 16, cy - 8), GREY, 0.36)
    for i, word in enumerate((st["drift_text"] if ok else "?").split(" + ")):
        text(img, word, (zx + 16, cy + 8 + i * 14), WHITE, 0.4)
    t = st["trim"]
    text(img, f"TRIM  X{t['x']:+5.1f}  Y{t['y']:+5.1f}  Z{t['z']:+5.1f}",
         (x0 + PANEL_W // 2, y1 - 12), (255, 200, 120), 0.4, 1, "c")


def action_banner(img, st):
    flag, left = st["action"]
    if not flag:
        return
    label, color = ACTION_STYLE[flag]
    boxed(img, label, (W // 2, 72), color, 0.9, 2)
    w = 160
    cv2.rectangle(img, (W // 2 - w // 2, 86), (W // 2 - w // 2 + round(w * left), 90), color, -1)


def draw(img, st):
    if not st["waiting"]:
        crosshair(img)
    else:
        dots = "." * (int(st["t"] * 2) % 4)
        text(img, f"CONNECTING TO DRONE{dots}", (W // 2 - 150, H // 2 - 40), WHITE, 0.8, 2)
        text(img, "join the drone's Wi-Fi if you haven't", (W // 2, H // 2 - 12), GREY, 0.5, 1, "c")
    top_bar(img, st)
    if st["fly"]:
        stick_panel(img, st["levels"], st["trim"])
        drift_panel(img, st)
        if not st["focused"]:
            boxed(img, "CLICK THIS WINDOW TO CONTROL", (W // 2, H // 2 + 70), YELLOW, 0.7, 2)
    keys_panel(img, st)
    action_banner(img, st)
    for i, msg in enumerate(st["toasts"]):
        boxed(img, msg, (W // 2, 135 + i * 34), WHITE, 0.55, 1, 8)
