"""Self-test: simulated drones on localhost exercise the real client code."""
import io, random, socket, struct, sys, threading, time
from PIL import Image, ImageDraw
import protocol, drone
from protocol import Sticks, control_frame, a_control_packet, b_parse, T_ACK

# ---- 1. control frame matches the app's Java formula -----------------------
s = Sticks(roll=178, pitch=60, throttle=200, yaw=120, flags=protocol.FLAG_TAKEOFF)
f = control_frame(s)
assert f == bytes([0x66, 178, 60, 200, 128, 1, 178 ^ 60 ^ 200 ^ 128 ^ 1, 0x99]), f.hex()
assert a_control_packet(Sticks())[:1] == b"\x03" and len(a_control_packet(Sticks())) == 9
print("control frame OK:", f.hex(" "))

# ---- 2. make real JPEG frames and strip the header like the drone does --------
def scan_data(n):
    img = Image.new("RGB", (640, 360), (20, 40, 90))
    d = ImageDraw.Draw(img)
    d.rectangle([40 + n * 30, 100, 200 + n * 30, 260], fill=(250, 200, 0))
    d.text((20, 20), f"frame {n}", fill=(255, 255, 255))
    b = io.BytesIO(); img.save(b, "JPEG", quality=50, subsampling=0); raw = b.getvalue()
    hdr = protocol.jpeg_header(50)
    sos = raw.index(b"\xff\xda"); body = raw[sos + 14:-2]
    assert raw[sos:sos + 14] == hdr[-14:]                       # same scan layout
    rebuilt = hdr + body + b"\xff\xd9"                          # what the app would show
    import numpy as np
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(rebuilt))), np.asarray(Image.open(io.BytesIO(raw)))), \
        "header tables differ from libjpeg's"
    return body, rebuilt

# ---- 3. fake Variant B drone ---------------------------------------------------
class FakeDrone(threading.Thread):
    def __init__(self, frames, drop=0.2):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.01); self.port = self.sock.getsockname()[1]
        self.frames, self.drop, self.peer = frames, drop, None
        self.commands, self.hellos, self.resent = [], 0, 0
    def frag(self, fid, idx, body):
        cnt = (len(body) + 1023) // 1024
        hdr = bytearray(0x38); hdr[0] = 0x93; hdr[1] = 1
        struct.pack_into("<Q", hdr, 8, fid); struct.pack_into("<Q", hdr, 0x18, max(0, fid - 4))
        struct.pack_into("<III", hdr, 0x20, idx, cnt, len(body)); struct.pack_into("<HH", hdr, 0x2C, 640, 360)
        hdr[0x30] = 50
        pkt = hdr + body[idx * 1024:(idx + 1) * 1024]; struct.pack_into("<H", pkt, 2, len(pkt)); return bytes(pkt)
    def run(self):
        while self.peer is None:                         # wait for hello
            try:
                d, a = self.sock.recvfrom(64)
                if d == protocol.B_HELLO: self.hellos += 1; self.peer = a
            except socket.timeout: pass
        rng = random.Random(1)
        for fid, body in enumerate(self.frames, start=1):
            cnt = (len(body) + 1023) // 1024
            for i in range(cnt):
                if rng.random() > self.drop: self.sock.sendto(self.frag(fid, i, body), self.peer)
            deadline = time.time() + 2
            while time.time() < deadline:               # resend what the acks say is missing
                try: d, _ = self.sock.recvfrom(4096)
                except socket.timeout: continue
                if d[:2] != b"\xef\x02": continue
                n = d[8]; seq, = struct.unpack_from("<I", d, 0x0C)
                if seq: self.commands.append(d[0x10:0x18])
                off, done = 0x20, False
                for _ in range(n):
                    efid, st, ln = struct.unpack_from("<QII", d, off)
                    if efid == fid and st == 1: done = True
                    if efid == fid and st == 0:
                        words = struct.unpack_from(f"<{(ln-16)//4}I", d, off + 16)
                        for i in range(cnt):
                            if not (words[i >> 5] >> (i & 31)) & 1:
                                self.sock.sendto(self.frag(fid, i, body), self.peer); self.resent += 1
                    off += ln
                if done: break
        end = time.time() + 1.0
        while time.time() < end:                         # keep listening for control
            try:
                d, _ = self.sock.recvfrom(4096)
                if d[:2] == b"\xef\x02" and struct.unpack_from("<I", d, 0x0C)[0]: self.commands.append(d[0x10:0x18])
            except socket.timeout: pass

bodies, originals = zip(*[scan_data(n) for n in range(5)])
fake = FakeDrone(list(bodies)); fake.start()
drone.B_IP, drone.B_PORT = "127.0.0.1", fake.port
link = drone.Link("b", fly=True, save_dir=None, verbose=False)
got = []
orig_on = link.rx.on_frag
def capture(fr, _o=orig_on):
    j = _o(fr)
    if j: got.append(j)
    return j
link.rx.on_frag = capture
link.pilot.pulse(protocol.FLAG_TAKEOFF)
t = threading.Thread(target=link.run_b, daemon=True); t.start()
fake.join(timeout=20); link.running = False; t.join(timeout=2)

print(f"hellos={fake.hellos} frames delivered={len(got)}/5 fragments resent={fake.resent}")
assert len(got) == 5, "not all frames reassembled"
for g, o in zip(got, originals):
    assert g == o, "reassembled JPEG differs from original"
    assert Image.open(io.BytesIO(g)).size == (640, 360)
print("all 5 frames byte-identical to the originals and decode as 640x360")
cmds = [c for c in fake.commands if c[:1] == b"\x66"]
assert fake.commands and fake.commands[0][:1] == b"\x64", "start command (0x64) not sent first"
assert cmds and cmds[0][5] & protocol.FLAG_TAKEOFF, "takeoff flag missing"
print(f"start cmd seen, {len(cmds)} control frames carried in acks, e.g. {cmds[0].hex(' ')}")

# ---- 4. Variant A over UDP --------------------------------------------------------
srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); srv.bind(("127.0.0.1", 0)); srv.settimeout(0.05)
drone.A_IP, drone.A_CTRL_PORT = "127.0.0.1", srv.getsockname()[1]
la = drone.Link("a", fly=True, save_dir=None, verbose=False)
ta = threading.Thread(target=la.run_a, daemon=True); ta.start()
seen = []; end = time.time() + 0.6
while time.time() < end:
    try: d, a = srv.recvfrom(64); seen.append(d)
    except socket.timeout: pass
srv.sendto(b"\x01\x00M\x07\x00", a); time.sleep(0.1)
la.running = False; ta.join(1)
try:
    while True: seen.append(srv.recvfrom(64)[0])
except socket.timeout: pass
assert seen[0] == b"\x01\x01" and any(p[:2] == b"\x03\x66" and len(p) == 9 for p in seen)
assert b"\x09\x01" in seen and seen[-1] == b"\x08\x01"
print(f"variant A: heartbeat, {sum(p[:1]==bytes([3]) for p in seen)} control packets, photo-button ack, control-off on exit")
# ---- depth: sampling, filtering, geometry and calibration maths (no model needed) ----------
import math, os, tempfile
import numpy as np
import depth, obstacles
cam = depth.Camera(hfov=60.0)
assert abs(cam.fx - depth.TRAIN_FOCAL) < 0.01, cam.fx            # a 60 deg lens = the training camera
assert abs(depth.Camera(hfov=90.0).fx - 0.5) < 1e-9
x, y, z = cam.point(0.5 + 0.5, 0.5, 2.0)                          # right edge of a 60 deg view, 2 m away
assert abs(math.degrees(math.atan2(x, z)) - 30.0) < 0.5 and y == 0.0
scene = np.full((252, 420), 3.0, np.float32)                     # a wall 3 m away ...
scene[100:200, 150:270] = 1.2                                     # ... with a chair 1.2 m away in front
dm = depth.DepthMap(scene, 0.0, cam)
assert abs(dm.box((0.33, 0.37, 0.67, 0.82)) - 1.2) < 0.01         # its box: the chair, not the wall
assert abs(dm.ahead() - 1.2) < 0.01 and abs(dm.at(0.05, 0.05) - 3.0) < 0.01
tr = obstacles.Track(1, (0.3, 0.3, 0.6, 0.8), 0.0, "chair", 0.9, True)
for z in (2.0, 2.04, 1.98, 3.5):                                  # one wild reading (3.5) is doubted ...
    tr.measure(z, 0.0)
assert 1.95 < tr.dist < 2.1 and tr.dist_doubt == 1, (tr.dist, tr.dist_doubt)
for z in (3.4, 3.45):                                             # ... but three in a row are believed
    tr.measure(z, 0.0)
assert abs(tr.dist - 3.45) < 1e-6
depth.CAL_FILE = os.path.join(tempfile.mkdtemp(), "depth_cal.json")   # never touch the real one
m = depth.DepthModel.__new__(depth.DepthModel)
m.camera, m.cal = depth.Camera(hfov=90.0), {"scale": 1.0, "samples": []}
m.raw = lambda frame: np.full((252, 420), 2.0, np.float32)        # model says 2 m (before lens)
scale = m.calibrate(None, 1.0)                                    # the wall is really 1 m away
assert abs(m.predict(None).mean() - 1.0) < 1e-4, scale
print(f"depth: near-surface sampling, jump doubting, 3D position, lens x{m.lens_factor:.2f} "
      f"and tape calibration (x{scale:.2f}) all check out")
print("ALL TESTS PASSED")
