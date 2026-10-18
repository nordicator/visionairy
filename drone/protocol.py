"""Sky Rider Cruise drone protocol: packet builders and parsers.

Recovered from the Android app com.cooingdv.skyridercurise v1.6.0.
The app speaks to two hardware variants and runs both stacks at once;
whichever drone answers wins.

  Variant A ("RTSP"): drone at 192.168.1.1
      video   rtsp://192.168.1.1:7070/webcam
      control UDP 192.168.1.1:7099

  Variant B ("MJPEG/MCU"): drone at 192.168.169.1
      video + control multiplexed on UDP 192.168.169.1:8800
"""
import struct
from dataclasses import dataclass, field

from jpeg_header import jpeg_header

# --------------------------------------------------------------------------
# Shared: the 8-byte flight-control frame (both variants)
# --------------------------------------------------------------------------
CENTER = 128

FLAG_TAKEOFF = 0x01      # "fast fly"  – app holds it for 1 s then clears
FLAG_LAND = 0x02         # "fast drop" – held 1 s
FLAG_STOP = 0x04         # emergency stop (motors off) – held 1 s
FLAG_FLIP = 0x08         # "circle turn end" – defined but never set by the app
FLAG_HEADLESS = 0x10     # headless mode – toggled, stays set
FLAG_GYRO_CAL = 0x80     # gyro calibration – held 2 s

SPEED_RANGES = {1: 50, 2: 60, 3: 127}   # right-stick half-range per speed mode


@dataclass
class Sticks:
    roll: int = CENTER       # byte1  right stick X  (right = higher)
    pitch: int = CENTER      # byte2  right stick Y  (forward = higher)
    throttle: int = CENTER   # byte3  left stick Y   (up = higher). 128 = hold altitude
    yaw: int = CENTER        # byte4  left stick X   (right/clockwise = higher)
    flags: int = 0           # byte5  bit flags above


def control_frame(s: Sticks) -> bytes:
    """66 RR PP TT YY FF CK 99, exactly as FlyController.FlyControlTask builds it."""
    roll = min(255, max(1, s.roll))
    pitch = min(255, max(1, s.pitch))
    yaw = s.yaw
    if 104 <= yaw <= 152:          # yaw dead-band
        yaw = CENTER
    yaw = min(255, max(1, yaw))
    thr = 0 if s.throttle == 1 else min(255, max(0, s.throttle))
    flags = s.flags & 0xFF
    ck = roll ^ pitch ^ thr ^ yaw ^ flags
    return bytes([0x66, roll, pitch, thr, yaw, flags, ck, 0x99])


# --------------------------------------------------------------------------
# Variant A – plain UDP to 192.168.1.1:7099
# --------------------------------------------------------------------------
A_IP, A_CTRL_PORT = "192.168.1.1", 7099
A_RTSP = "rtsp://192.168.1.1:7070/webcam"

A_HEARTBEAT = b"\x01\x01"          # every 1 s, starts once video plays
A_CONTROL_OFF = b"\x08\x01"        # sent when the on-screen sticks are hidden
A_CAMERA_1 = b"\x06\x01"           # switch camera (toggle 06 02 / 06 01)
A_CAMERA_2 = b"\x06\x02"
A_ACK_PHOTO_BTN = b"\x09\x01"      # ack after drone reports its photo button
A_ACK_VIDEO_BTN = b"\x09\x02"      # ack after drone reports its video button


def a_control_packet(s: Sticks) -> bytes:
    """Variant A wraps the 8-byte frame with a 0x03 prefix. Sent every 50 ms."""
    return b"\x03" + control_frame(s)


def a_parse_status(pkt: bytes) -> dict:
    """Drone -> app on port 7099 (<=20 bytes)."""
    out = {"raw": pkt.hex(" ")}
    if pkt:
        out["device_id"] = pkt[0]              # used to pick a 'fake resolution'
    if len(pkt) >= 2:
        out["camera_switch_needs_restart"] = pkt[1] == 1
    if len(pkt) > 4 and pkt[0] != 5:
        if pkt[2] == ord("M"):
            out["photo_button_count"] = pkt[3]
        elif pkt[2] == ord("X"):
            out["video_button_count"] = pkt[4]
    return out


# --------------------------------------------------------------------------
# Variant B – custom protocol on UDP 192.168.169.1:8800
#   app -> drone packets start with 0xEF, drone -> app with 0x93.
#   Byte 1 is the type, bytes 2-3 the total length (little-endian).
# --------------------------------------------------------------------------
B_IP, B_PORT = "192.168.169.1", 8800

B_HELLO = b"\xef\x00\x04\x00"       # every ~100 ms until video starts
B_CMD_START = b"\x64"              # sent once, after the first video frame
B_CMD_CONTROL_OFF = b"\x65"        # sticks hidden (Variant A uses 08 01)

T_FRAG, T_ACK, T_CTLMSG, T_QUERY_RESP = 1, 2, 4, 8
FRAG_SIZE = 1024
SLOTS = 4


@dataclass
class Frag:
    frame_id: int
    oldest_kept: int      # drone's oldest frame still buffered
    index: int
    count: int
    size: int             # total scan-data bytes in the frame
    width: int
    height: int
    quality: int          # 5/10/25/50/75/100 -> picks the JPEG header
    data: bytes


def b_parse(pkt: bytes):
    """Return (type, parsed) for a drone->app packet, or (None, None)."""
    if len(pkt) < 4 or pkt[0] != 0x93:
        return None, None
    if struct.unpack_from("<H", pkt, 2)[0] != len(pkt):
        return None, None
    t = pkt[1]
    if t == T_FRAG and len(pkt) >= 0x38:
        fid, = struct.unpack_from("<Q", pkt, 0x08)
        oldest, = struct.unpack_from("<Q", pkt, 0x18)
        idx, cnt, size = struct.unpack_from("<III", pkt, 0x20)
        w, h = struct.unpack_from("<HH", pkt, 0x2C)
        q = pkt[0x30]
        return t, Frag(fid, oldest, idx, cnt, size, w, h, q, pkt[0x38:])
    if t == T_ACK and len(pkt) >= 8:
        return t, struct.unpack_from("<I", pkt, 4)[0]           # acked command seq
    if t == T_CTLMSG and len(pkt) >= 12:
        seq, = struct.unpack_from("<I", pkt, 4)
        n, = struct.unpack_from("<H", pkt, 8)
        return t, (seq, pkt[12:12 + n])                         # 'M'/'X' button events etc.
    if t == T_QUERY_RESP:
        return t, pkt
    return t, pkt


def b_ctlmsg_reply(seq: int) -> bytes:
    """App must answer each type-4 message with EF 08 08 00 <seq u32>."""
    return b"\xef\x08\x08\x00" + struct.pack("<I", seq)


@dataclass
class Slot:
    state: int = 0        # 0 empty, 1 receiving, 2 complete, 3 delivered
    frame_id: int = 0
    count: int = 0
    size: int = 0
    width: int = 640
    height: int = 360
    quality: int = 50
    bitmap: list = field(default_factory=list)
    buf: bytearray = field(default_factory=bytearray)


class BReceiver:
    """Reassembles video frames and builds the ack packets the drone expects."""

    def __init__(self):
        self.slots = [Slot() for _ in range(SLOTS)]
        self.max_seen = 0
        self.oldest_kept = 0
        self.next_expected = 1
        self.active_slot = 0
        self.connected = False
        self.cmd_seq = 0
        self.cmd = b""
        self.qpara = b"\x00\x00\x00\x00"   # never set by the app
        self.camera = 0

    # ---- incoming ------------------------------------------------------
    def on_frag(self, f: Frag):
        """Feed a fragment. Returns a complete JPEG (bytes) or None."""
        if f.index + 1 > f.count:
            return None
        self.oldest_kept = f.oldest_kept
        self.max_seen = max(self.max_seen, f.frame_id)
        si = (f.frame_id + 3) & 3
        self.active_slot = si
        s = self.slots[si]
        if s.state in (0, 3, 4):
            if not (s.frame_id == f.frame_id - 4 or (f.frame_id <= 4 and s.frame_id == 0)):
                return None
            self.connected = True
            hdr = jpeg_header(f.quality, f.width, f.height)
            s.state, s.frame_id, s.count, s.size = 1, f.frame_id, f.count, f.size
            s.width, s.height, s.quality = f.width, f.height, f.quality
            s.bitmap = [0] * ((f.count + 31) // 32)
            s.buf = bytearray(hdr) + bytearray(f.size) + b"\xff\xd9"
        elif s.state != 1 or s.frame_id != f.frame_id:
            return None
        s.bitmap[f.index >> 5] |= 1 << (f.index & 31)
        hlen = len(s.buf) - s.size - 2
        start = f.index * FRAG_SIZE
        n = min(FRAG_SIZE, s.size - start, len(f.data))
        s.buf[hlen + start:hlen + start + n] = f.data[:n]
        if all((s.bitmap[i >> 5] >> (i & 31)) & 1 for i in range(s.count)):
            s.state = 2
            if s.frame_id >= self.next_expected:      # app waits strictly in order;
                s.state = 3                           # we skip ahead instead of stalling
                self.next_expected = s.frame_id + 1
                return bytes(s.buf)
        return None

    # ---- outgoing ------------------------------------------------------
    def set_command(self, payload: bytes):
        """Queue a command (<=8 bytes). It rides on every ack until replaced."""
        assert len(payload) <= 8
        self.cmd_seq += 1
        self.cmd = payload

    def build_ack(self, idle: bool) -> bytes:
        """idle=True when no fragment arrived in the last ~20 ms."""
        entries, n = bytearray(), 0
        for i, s in enumerate(self.slots):
            if not idle and i == self.active_slot:
                continue
            abandon = (self.max_seen - s.frame_id) >= 5 or (s.state == 0 and self.max_seen > i + 1)
            if abandon:
                fid = i + 1 if s.frame_id == 0 else s.frame_id + 4
                entries += struct.pack("<QII", fid, 3, 16)
                n += 1
                continue
            if s.frame_id <= self.oldest_kept or s.state == 0:
                continue
            status = {1: 0, 2: 1, 3: 1, 4: 2}.get(s.state, 0xFFFFFFFF)
            bm = struct.pack(f"<{len(s.bitmap)}I", *s.bitmap)
            entries += struct.pack("<QII", s.frame_id, status, 16 + len(bm)) + bm
            n += 1
        if idle:
            ok = True
            off = 0
            for _ in range(n):
                st, ln = struct.unpack_from("<II", entries, off + 8)
                if st not in (1, 2):
                    ok = False
                    break
                off += ln
            if ok:
                entries += struct.pack("<QII", self.next_expected, 3, 16)
                n += 1
        body = bytearray(0x20)
        body[0:2] = b"\xef\x02"
        body[4:8] = b"\x02\x02\x00\x01"
        body[8] = n
        if self.cmd_seq:
            struct.pack_into("<I", body, 0x0C, self.cmd_seq)
            body[0x10:0x10 + len(self.cmd)] = self.cmd
        body[0x18:0x1C] = self.qpara
        body[0x1C] = self.camera
        pkt = body + entries + bytes(8)        # the app always sends 8 trailing zero bytes
        struct.pack_into("<H", pkt, 2, len(pkt))
        return bytes(pkt)
