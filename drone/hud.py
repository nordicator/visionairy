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
# one colour per kind of object (BGR). Red and yellow are kept for warnings, so none use them.
OBJECT_COLORS = {"human": (255, 90, 255),     # magenta
                 "chair": (255, 190, 40),     # sky blue
                 "table": (90, 220, 90),      # green
                 "tv": (255, 140, 190),       # purple
                 "bag": (0, 150, 255),        # orange
                 "box": (120, 190, 220),      # tan
                 "wall": (215, 215, 215),     # light grey
                 "obstacle": RED}

ACTION_STYLE = {1: ("TAKING OFF", GREEN), 2: ("LANDING", YELLOW), 4: ("MOTORS OFF", RED),
                0x80: ("CALIBRATING GYRO", YELLOW)}


def letterbox(shape):
    """(x, y, width, height) of a frame of this shape on the canvas."""
    h, w = shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    return (W - nw) // 2, (H - nh) // 2, nw, nh


def fit(frame):
    """Letterbox a video frame (or nothing) onto the W x H canvas."""
    canvas = np.zeros((H, W, 3), np.uint8)
    if frame is None:
        return canvas
    x, y, nw, nh = letterbox(frame.shape)
    canvas[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def to_frame(cx, cy, shape):
    """Canvas pixel -> (u, v) fractions of the video frame, or None if outside the picture."""
    x, y, nw, nh = letterbox(shape)
    u, v = (cx - x) / nw, (cy - y) / nh
    return (u, v) if 0 <= u < 1 and 0 <= v < 1 else None


SPEED_BUTTONS = {s: (66 + (s - 1) * 34, H - 40, 66 + (s - 1) * 34 + 28, H - 16) for s in (1, 2, 3)}


def speed_button_at(x, y):
    """Which speed button (1-3) is at canvas pixel (x, y), or None."""
    for s, (x0, y0, x1, y1) in SPEED_BUTTONS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return s
    return None


def speed_buttons(img, speed):
    panel(img, 12, H - 46, 66 + 3 * 34, H - 10)
    text(img, "SPEED", (20, H - 23), GREY, 0.4)
    for s, (x0, y0, x1, y1) in SPEED_BUTTONS.items():
        keycap(img, x0, y0, x1 - x0, y1 - y0, str(s), s == speed)


def goto_marker(img, st):
    """Ring on the click-to-go target, with a line from the crosshair it's being flown to."""
    x, y, nw, nh = letterbox(st["frame_shape"])
    u, v = st["goto"]
    p = (x + round(u * nw), y + round(v * nh))
    cv2.line(img, (W // 2, H // 2), p, ACCENT, 1, cv2.LINE_AA)
    r = 14 + round(3 * np.sin(st["t"] * 6))
    cv2.circle(img, p, r, ACCENT, 2, cv2.LINE_AA)
    cv2.circle(img, p, 3, ACCENT, -1, cv2.LINE_AA)
    text(img, "GO TO", (p[0], p[1] - r - 6), ACCENT, 0.45, 1, "c")


def object_color(label):
    return OBJECT_COLORS.get(label, WHITE)


def dim(color, k=0.55):
    return tuple(round(c * k) for c in color)


def object_boxes(img, st):
    """Corner brackets on each tracked object in its type's colour. A threat gets thick red
    brackets; closing in shows seconds-to-contact in yellow; briefly lost = thin and dim."""
    x, y, nw, nh = letterbox(st["frame_shape"])
    threat = False
    tags = []
    # threats last, so their brackets and names end up on top
    for o in sorted(st["objects"], key=lambda o: (o["threat"], o["ttc"] is not None)):
        x0, y0, x1, y1 = (round(x + o["box"][0] * nw), round(y + o["box"][1] * nh),
                          round(x + o["box"][2] * nw), round(y + o["box"][3] * nh))
        color, thick = object_color(o["label"]), 2
        if o["threat"]:
            color, thick = RED, 3
        elif o["lost"]:
            color, thick = dim(color), 1
        threat |= o["threat"]
        c = max(6, min(x1 - x0, y1 - y0) // 4)              # bracket arm length
        for px, py, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(img, (px, py), (px + c * dx, py), color, thick, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + c * dy), color, thick, cv2.LINE_AA)
        tags.append((o, x0, y0, color))
    for o, x0, y0, color in tags:                           # names over every bracket
        name = f"{o['label'].upper()} #{o['id']}"
        if o.get("dist") is not None:                       # ~ = predicted, not just measured
            name += f"  {'' if o.get('dist_fresh') else '~'}{o['dist']:.1f} m"
        ttc = f"  {o['ttc']:.1f}s" if o["ttc"] else ""
        (tw, th), _ = cv2.getTextSize(name + ttc, FONT, 0.5, 1)
        ty = max(y0 - 6, 50)                                # baseline, kept below the top bar
        tx = max(0, min(x0, W - tw - 8))
        panel(img, tx, ty - th - 5, min(W, tx + tw + 8), ty + 5, 0.7)
        cv2.rectangle(img, (tx, ty - th - 5), (tx + 2, ty + 5), color, -1)   # colour chip
        nx = tx + 6 + text(img, name, (tx + 6, ty), color, 0.5)
        if ttc:
            text(img, ttc, (nx, ty), RED if o["threat"] else YELLOW, 0.5)
    ahead, close = st.get("ahead") or (None, False)
    if (threat or close) and int(st["t"] * 4) % 2:
        boxed(img, "OBSTACLE AHEAD" + (f"  {ahead:.1f} m" if ahead is not None else ""), (W // 2, 104), RED, 0.7, 2)


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


def ahead_readout(img, st):
    """Distance to the nearest thing straight ahead, under the crosshair."""
    ahead, close = st.get("ahead") or (None, False)
    if ahead is None:
        return
    color = RED if close else YELLOW if ahead < 1.5 else WHITE
    s = f"{ahead:.1f} m"
    (tw, th), _ = cv2.getTextSize(s, FONT, 0.5, 1)
    panel(img, W // 2 - tw // 2 - 6, H // 2 + 22, W // 2 + tw // 2 + 6, H // 2 + 44, 0.6)
    text(img, s, (W // 2, H // 2 + 38), color, 0.5, 1, "c")


def depth_range():
    import depth
    return depth.VIEW_NEAR, depth.VIEW_FAR


def depth_view(img, st):
    """The depth map over the video (near = red, far = blue), with a scale."""
    x, y, nw, nh = letterbox(st["frame_shape"])
    colour = cv2.resize(st["depth_img"], (nw, nh))
    img[y:y + nh, x:x + nw] = cv2.addWeighted(img[y:y + nh, x:x + nw], 0.35, colour, 0.65, 0)
    near, far = depth_range()
    y0 = 36                                              # under the top bar
    panel(img, W // 2 - 120, y0, W // 2 + 120, y0 + 30)
    for i in range(200):
        c = cv2.applyColorMap(np.array([[255 - int(i * 255 / 199)]], np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        cv2.line(img, (W // 2 - 100 + i, y0 + 5), (W // 2 - 100 + i, y0 + 13), tuple(int(v) for v in c), 1)
    for m in (near, 1.0, 2.0, far):                      # log scale, like DepthMap.colour
        x = W // 2 - 100 + round(200 * (np.log(m) - np.log(near)) / (np.log(far) - np.log(near)))
        text(img, f"{m:g} m", (x, y0 + 26), WHITE, 0.35, 1, "c")


def signal_bars(img, x, y, pct):
    for i in range(4):
        h = 4 + i * 3
        on = pct is not None and pct > i * 25
        cv2.rectangle(img, (x + i * 5, y - h), (x + i * 5 + 3, y), GREEN if on else (90, 90, 90), -1)


def top_bar(img, st):
    panel(img, 0, 0, W, 30)
    label, color = st["status"]
    if label != "SIGNAL LOST" or int(st["t"] * 3) % 2:
        cv2.circle(img, (16, 15), 6, color, -1, cv2.LINE_AA)
    x = 30 + text(img, label, (30, 20), color, 0.5, 1)
    if st["wifi"]:
        ssid, sig = st["wifi"]
        signal_bars(img, x + 16, 21, sig)
        if sig is not None:
            text(img, f"{sig}%", (x + 40, 20), GREY, 0.45)

    right = W - 12
    items = []
    if st["rec"] is not None:
        m, sec = divmod(int(st["rec"]), 60)
        items.append((f"REC {m:02d}:{sec:02d}", RED))
    if st["flight"] is not None:
        m, sec = divmod(int(st["flight"]), 60)
        items.append((f"{m:02d}:{sec:02d}", WHITE))
    items.append((f"{st['fps']:.0f} fps", GREY))
    if st["age_ms"] is not None:
        a = st["age_ms"]
        items.append((f"{a} ms", GREEN if a < 300 else YELLOW if a < 1000 else RED))
    # the models only get a word in when something's wrong with them
    for key in ("model", "depth_status"):
        if st.get("objects") is not None and st.get(key) and not st[key][1]:
            items.append((st[key][0], YELLOW))
    for item, c in items:
        right -= text(img, item, (right, 20), c, 0.45, 1, "r") + 18
        if item.startswith("REC") and int(st["t"] * 2) % 2:
            cv2.circle(img, (right + 6, 15), 5, RED, -1, cv2.LINE_AA)
            right -= 14


def chips(img, st):
    """Modes that change how the drone behaves, as small labels in the middle of the top bar."""
    out = []
    if st.get("webcam"):
        out.append(("WEBCAM TEST", ACCENT))
    elif not st["fly"]:
        out.append(("PROBE - NO FLIGHT COMMANDS", YELLOW))
    if st.get("headless"):
        out.append(("HEADLESS", ACCENT))
    if st["fly"] and st["auto"] == "calibrating":
        out.append(("CALIBRATING DRIFT" + "." * (int(st["t"] * 3) % 4), YELLOW))
    elif st["fly"] and st["auto"] == "off":
        out.append(("AUTO-TRIM OFF", GREY))
    widths = [cv2.getTextSize(t, FONT, 0.45, 1)[0][0] + 20 for t, _ in out]
    x = W // 2 - (sum(widths) + 8 * (len(out) - 1)) // 2
    for (t, c), w in zip(out, widths):
        cv2.rectangle(img, (x, 5), (x + w, 25), c, 1, cv2.LINE_AA)
        text(img, t, (x + 10, 20), c, 0.45)
        x += w + 8


HELP_ROWS = (("W A S D", "move"), ("UP / DOWN  (R / F)", "climb / descend"), ("T", "take off"),
             ("L", "land"), ("SPACE", "motors off"), ("C", "gyro calibrate (on the ground)"),
             ("H", "headless mode on / off"), ("SHIFT + move key", "nudge trim"),
             ("K", "auto-trim on / off"), ("BACKSPACE", "reset trim"), ("CLICK", "fly to that spot"),
             ("G", "follow the throw path (iOS app tracks the drone)"),
             ("P / V", "photo / record"), ("O / Z", "object boxes / depth view"),
             ("M", "depth calibration (wall at --cal-distance)"), ("?", "hide this"), ("ESC", "quit"))


def help_panel(img):
    w, row = 480, 22
    h = 20 + row * len(HELP_ROWS)
    x0, y0 = W // 2 - w // 2, H // 2 - h // 2
    panel(img, x0, y0, x0 + w, y0 + h, 0.8)
    for i, (k, what) in enumerate(HELP_ROWS):
        y = y0 + 26 + i * row
        text(img, k, (x0 + 16, y), ACCENT, 0.45)
        text(img, what, (x0 + 190, y), WHITE, 0.45)


def action_banner(img, st):
    flag, left = st["action"]
    if not flag:
        return
    label, color = ACTION_STYLE[flag]
    boxed(img, label, (W // 2, 72), color, 0.9, 2)
    w = 160
    cv2.rectangle(img, (W // 2 - w // 2, 86), (W // 2 - w // 2 + round(w * left), 90), color, -1)


def draw(img, st):
    if st.get("depth_img") is not None:
        depth_view(img, st)
    if not st["waiting"]:
        crosshair(img)
        ahead_readout(img, st)
    else:
        dots = "." * (int(st["t"] * 2) % 4)
        text(img, f"CONNECTING TO DRONE{dots}", (W // 2 - 150, H // 2 - 40), WHITE, 0.8, 2)
        text(img, "starting the webcam" if st.get("webcam") else "join the drone's Wi-Fi if you haven't",
             (W // 2, H // 2 - 12), GREY, 0.5, 1, "c")
    top_bar(img, st)
    if st["objects"]:
        object_boxes(img, st)
    if st["goto"]:
        goto_marker(img, st)
    chips(img, st)
    if st["fly"]:
        speed_buttons(img, st["speed"])
        if not st["focused"]:
            boxed(img, "CLICK THIS WINDOW TO CONTROL", (W // 2, H // 2 + 70), YELLOW, 0.7, 2)
    action_banner(img, st)
    for i, msg in enumerate(st["toasts"]):
        boxed(img, msg, (W // 2, 135 + i * 34), WHITE, 0.55, 1, 8)
    if st["show_keys"]:
        help_panel(img)
    else:                                    # beside the speed buttons, out of the boxes' way
        x = 12 + (66 + 3 * 34 - 12) + 10 if st["fly"] else 12
        panel(img, x, H - 46, x + 66, H - 10)
        text(img, "? keys", (x + 10, H - 23), GREY, 0.4)
