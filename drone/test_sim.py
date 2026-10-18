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
print("ALL TESTS PASSED")
