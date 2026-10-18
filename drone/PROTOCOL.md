# Sky Rider Cruise drone – Wi-Fi protocol

Recovered from the Android app `com.cooingdv.skyridercurise` v1.6.0 (jadx for the Java code, llvm-objdump for `libmjpeg_jni.so`).
All multi-byte numbers are little-endian unless noted.

The app supports **two hardware variants** and starts both network stacks at launch. Whichever drone answers wins. You can tell which one you have from the IP your phone gets after joining the drone's Wi-Fi (SSIDs usually start with `HD-720P-`, `HD-FPV-`, `HD720-` or `FHD-`):

| Your IP | Variant | Drone | Video | Control |
|---|---|---|---|---|
| 192.168.1.x | **A** | 192.168.1.1 | RTSP `rtsp://192.168.1.1:7070/webcam` | UDP → 192.168.1.1:7099 |
| 192.168.169.x | **B** | 192.168.169.1 | custom UDP on :8800 | same socket, :8800 |

---

## 1. Flight-control frame (both variants)

8 bytes, sent every **50 ms** while the on-screen sticks are visible:

| Byte | Meaning | Range |
|---|---|---|
| 0 | `0x66` start marker | |
| 1 | **Roll**, right stick X (right = higher) | 1–255, center 128 |
| 2 | **Pitch**, right stick Y (forward = higher) | 1–255, center 128 |
| 3 | **Throttle**, left stick Y (up = higher) | 0–255. In altitude-hold mode (the app's default) 128 = hover. Otherwise 0 = idle |
| 4 | **Yaw**, left stick X (clockwise = higher) | 1–255, center 128. Values 104–152 are snapped to 128 |
| 5 | Flags (see below) | |
| 6 | Checksum = `b1 ^ b2 ^ b3 ^ b4 ^ b5` | |
| 7 | `0x99` end marker | |

**Flags (byte 5):**

| Bit | Value | Function | How the app uses it |
|---|---|---|---|
| 0 | 0x01 | Take off ("fast fly") | Set for 1 s, then cleared |
| 1 | 0x02 | Land ("fast drop") | Set for 1 s |
| 2 | 0x04 | Emergency stop | Set for 1 s |
| 3 | 0x08 | Flip / "circle turn" | Defined but never set by this app |
| 4 | 0x10 | Headless mode | Toggle, stays set |
| 7 | 0x80 | Gyro calibration | Set for 2 s |

**Stick scaling in the app.**
- **Left stick:** `128 ± 127 × deflection`.
- **Right stick:** `128 ± R × deflection`, where R depends on the speed mode: 50 (default), 60 or 127.
- **Trims:** the trim buttons add a step to the base value. Each step is 2, 3 or 4 depending on speed mode, up to ±12 steps for roll and pitch. Yaw trim uses step × 4.
- **Tilt mode:** the phone's accelerometer drives the right stick.

## 2. Variant A (192.168.1.1)

| Direction | Bytes | Meaning |
|---|---|---|
| app → 7099 | `01 01` | Heartbeat, every 1 s once video is playing |
| app → 7099 | `03` + 8-byte control frame | Flight control, every 50 ms |
| app → 7099 | `08 01` | Sticks hidden / control off |
| app → 7099 | `06 01` / `06 02` | Switch camera (toggle) |
| app → 7099 | `09 01` / `09 02` | Ack after the drone reports a photo / video button press |
| drone → app | `[0]`=device id, `[1]`=1 → reopen RTSP after camera switch, `[2]`=`'M'` → photo button (count in `[3]`), `[2]`=`'X'` → video button (count in `[4]`) | Status, ≤20 bytes |

Video is RTSP/MJPEG (VLC or ffplay can open it). The drone also serves files over HTTP (`http://192.168.1.1/PHOTO/O/…`, `/DCIM/…`) and FTP (user `ftp`, password `ftp`). This app never uses them: photos and videos are recorded on the phone from the live stream.

## 3. Variant B (192.168.169.1:8800)

The app binds one UDP socket on a random port and uses it for everything. Outbound packets start with `0xEF`, inbound with `0x93`. Byte 1 is the packet type and bytes 2–3 the total length.

### Connection
1. Send **hello** `EF 00 04 00` every ~100 ms until the first video fragment arrives.
2. After that, send an **ack** (type 2) every ~25 ms. Flight commands ride inside the acks.
3. If no fragment arrives for 3 s, treat the link as dropped and go back to step 1.
4. After the first complete frame, send command `64`. When the sticks are hidden, send `65`.

### Video fragment (drone → app, type 1)

| Offset | Size | Field |
|---|---|---|
| 0x00 | 1 | `0x93` |
| 0x01 | 1 | `0x01` |
| 0x02 | 2 | Packet length |
| 0x08 | 8 | Frame id (starts at 1, increments) |
| 0x18 | 8 | Oldest frame id the drone still buffers |
| 0x20 | 4 | Fragment index |
| 0x24 | 4 | Fragment count |
| 0x28 | 4 | Frame size: JPEG scan bytes, header not included |
| 0x2C | 2 | Width |
| 0x2E | 2 | Height |
| 0x30 | 1 | JPEG quality: 5 / 10 / 25 / 50 / 75 / 100 |
| 0x38 | ≤1024 | Payload. Fragment *i* goes at byte offset *i* × 1024 |

The drone sends **only the entropy-coded scan data**. To get a viewable JPEG, put a header in front and `FF D9` at the end. The header uses standard IJG quantisation tables scaled to the frame's quality, standard Huffman tables, 3 components and 1×1 sampling. `jpeg_header.py` regenerates all six variants byte-for-byte. The app keeps 4 frames in flight, in slot `(id + 3) & 3`.

### Ack + command (app → drone, type 2)

| Offset | Size | Field |
|---|---|---|
| 0x00 | 2 | `EF 02` |
| 0x02 | 2 | Total length |
| 0x04 | 4 | `02 02 00 01` |
| 0x08 | 1 | Number of entries |
| 0x0C | 4 | Command sequence number (0 = no command yet) |
| 0x10 | 8 | Command payload: the 8-byte control frame, `64`, or `65` |
| 0x18 | 4 | Quality parameters (always 0 in this app) |
| 0x1C | 1 | Camera index |
| 0x20 | … | Entries |
| end | 8 | Zero padding. The app always appends it |

Each entry is `u64 frame_id, u32 status, u32 entry_len`, followed by a bitmap of received fragments (1 bit per fragment, packed into u32 words):

| Status | Meaning |
|---|---|
| 0 | Still receiving |
| 1 | Complete |
| 2 | (unused state) |
| 3 | Give up on / skip this id. Entry length 16, no bitmap |

The drone resends any fragment whose bit is 0. The last command is repeated in every ack until a new one replaces it. The drone confirms commands with a type-2 packet carrying the sequence number at 0x04.

### Other inbound types
- **Type 4 (control message):** sequence number at 0x04, length at 0x08, payload at 0x0C. The payload is `77` ('M') for the photo button, `88` ('X') for the video button, otherwise a device id. **Each one must be answered with** `EF 08 08 00 <seq u32>`.
- **Type 8:** query-info response. The app never sends the matching query.

## 4. What still needs a real drone to confirm
- Exact stick direction conventions on your unit. They're inferred from the UI code, so check with the propellers removed.
- That throttle 128 really means "hold altitude" on your firmware.
- Variant B's give-up/skip handshake (status 3) under heavy packet loss. The test client skips ahead instead of stalling.
