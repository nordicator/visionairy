# VisionAiry: guide for the AI agent picking this up

Read this first. README.md is the user-facing version.

## What this is

An ESP32 "beacon" with an IMU is tracked in 3D **using only motion sensors (no camera tracking of the beacon)**. Its live position and trail are shown:
- in AR on an iPhone: native ARKit app in `ios/` (preferred), or the older web version at `/headset`;
- in a 3D graph on the laptop at `/graph` (three.js).

The user wants the beacon's position always displayed with a trailing line, as accurately as possible, and coordinate-based. Be upfront about the limits below instead of overpromising.

## Hardware

- **Freenove ESP32-WROOM** (CH340 USB serial, shows up as `/dev/cu.usbserial-*`). FQBN `esp32:esp32:esp32`.
- **IMU:** the user calls it an MPU-6050, but it reports WHO_AM_I `0x70` (MPU-6500). They're register-compatible.
  - VCC → 3V3, GND → GND, SDA → **GPIO32**, SCL → **GPIO33**. INT is not connected.
  - Everything is wired on one side of the board. Avoid GPIO12 (strapping pin) and GPIO34–39 (input only).
- Onboard LED is GPIO2. An optional external LED is on GPIO26.
- Phones: **iPhone 14 Pro on iOS 27** (paired with the old laptop, Developer Mode on). There's also an older iPhone 14 named "Awsaf".

## Layout

```
server.js            HTTPS :8443 (web pages, wss /ws) + plain ws :8080 (/ws, for the iOS app)
                     + beacon link: USB serial (preferred) or UDP broadcast :4210
public/headset.*     web AR (phone browser). Phone position is dead-reckoned; less accurate than ios/
public/graph.html    laptop 3D graph: beacon board + trail, phone frustum + trail, distance
public/index.html    laptop home: QR code to open /headset on the phone without typing
esp32/beacon/        firmware (beacon.ino). esp32/flash.sh compiles + uploads; reads esp32/.env
ios/                 native ARKit app (SwiftUI + SceneKit). project.yml (xcodegen) + build.sh
```

## Data flow and protocol

Beacon → server, one text line per message, over USB serial **115200** and/or UDP broadcast on port 4210:
- `VAS1 <ms> <px> <py> <pz> <qw> <qx> <qy> <qz> <flags>`: state, 30 Hz. Flags: 1 still, 2 airborne, 4 calibrated. q is board→world.
- `VAB1 <ms> <rssi> <led> <imu> <accel g> <spin dps> <airborne> <cal> <yaw> <pitch>`: heartbeat, 4 Hz.
- `VAE1 launch|land|cal ok|cal moving|zero ...`: events (format in the beacon.ino header).

Commands to the beacon, sent as serial or UDP text lines: `cal`, `zero`, `on|off|blink`, `imu` (toggles a debug stream).

The server turns these into WebSocket JSON (`bstate`, `beacon`, `imu`) for every client. Clients send `{type:'cmd', cmd}` and `{type:'phone', p, x, y, z}` (the phone's pose, relayed to `/graph`).

## Coordinate frame (shared by everything)

- Origin = the calibration spot. Z up, Y = the IMU's printed Y arrow at calibration (= the phone's top edge), X = Y × Z (right). Metres.
- **Sync (ios app):** lay the phone flat, screen up, with the beacon touching its right edge, tops lined up and the beacon's Y arrow parallel to the phone's top. Tap SYNC, then leave the beacon still for 5 s; the phone can move.
  - The phone's pose is captured at the tap; the origin is offset from the camera to the beacon's centre (`Tracker.beaconOffset`). The app sends `cal` 0.8 s before the 5 s are up, so the beacon averages its last 0.8 s of rest. On `cal ok` the app sets `graphRoot` there. If the beacon's spin goes above 12 °/s during the wait, the sync is cancelled.
  - The user found the old flow (beacon lying on the phone's screen, auto-calibrating after 2 s) awkward.
- **Re-zero:** touch the beacon to the phone and tap RE-ZERO. The beacon's position resets to 0 and the origin moves to the phone. This clears drift.

## How the beacon tracks (esp32/beacon/beacon.ino)

At 500 Hz: gyro integration plus a Mahony-style gravity correction when near 1 g gives orientation. Rotating the accelerometer into the world frame and subtracting g gives linear acceleration, which is integrated to velocity and then position. Two things keep the drift down:
- **Zero-velocity update:** after 80 ms still (|a_lin| < 1 m/s² and spin < 25 °/s), velocity is set to 0. While still, measured drift was 0.0 cm over 10 s.
- **Free fall** (|a| < 0.5 g for 30 ms): it switches to exact ballistics.

**Honest limit:** double-integrating a ~$3 IMU drifts while moving (cm to tens of cm after a few seconds). No library fixes this. Accurate continuous position needs an external reference such as UWB (DW3000), ultrasonic or Lighthouse. The user was told this and chose to go ahead with what they have. The old laptop's log showed questionable release velocities on tosses, so throw accuracy is unverified.

## Drone following a throw path

- iOS `DroneTracker.swift`: DRONE button, tap the drone; Vision tracks its box and LiDAR `sceneDepth` gives its depth, so it gets a 3D position in ARKit world. The app sends `{type:'drone', ok, world, path, p}`, where `p` is in the latest throw's path frame (origin = release spot, Z up, Y = throw direction) and `path` is that throw's `id`.
- `drone/follow.py` (G in `drone.py`): a stdlib WebSocket client to :8080, a heading probe (the drone never yaws, so it pushes forward once to learn its axes), then a PD follow at 0.25 m/s. It lands when done, or when the drone fix has been missing for 3 s.
- Tested only in simulation (`follow.py` against a lagged, rotated, drifting model: ~2 cm mean off-path). Not flown yet. The phone and laptop must share the drone's Wi-Fi; cheap drone APs may isolate clients.

## Setup on a new machine

1. `npm install` and `npm start`. On macOS, `serialport` loads its prebuilt binding even though npm blocks install scripts.
2. **Firmware:**
   - `brew install arduino-cli`
   - `arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json`
   - `arduino-cli core update-index && arduino-cli core install esp32:esp32`. The download is large (~3 GB unpacked), so check disk space first; the old laptop kept filling up.
   - Create `esp32/.env` from `.env.example` with `WIFI_SSID` / `WIFI_PASS`. It's gitignored; ask the user for the Wi-Fi.
   - `esp32/flash.sh`. **Stop the server first**, because it holds the serial port.
3. **iOS app:**
   - `brew install xcodegen`
   - `ios/build.sh` builds, installs on the connected iPhone and launches. It bakes the laptop's en0 IP in as `VA_SERVER_HOST`.
   - Building from the Xcode GUI leaves that host at 127.0.0.1 unless you set the `VA_SERVER_HOST` build setting.
   - Signing: team `464MGJA3HD` (paid account, "Apple Development: Awsaf Syed"). The new laptop needs that Apple ID signed in to Xcode (Settings → Accounts).
   - The phone must be paired (`xcrun devicectl manage pair --device <id>`) and have Developer Mode on.

## Known issues and gotchas

- **Unresolved at handoff:** the iOS app installed but iOS 27 refused to launch it: "profile has not been explicitly trusted by the user".
  - Earlier the phone said an internet connection was needed to verify the developer. The user then couldn't find the trust entry in Settings (iOS 27 may have moved it from General → VPN & Device Management).
  - The profile does include the phone (UDID 00008120-000E703A3442201E). Next step was running from the Xcode GUI.
- **Serial above 115200 corrupts** on this board, and uploads at 921600 fail. flash.sh uploads at 115200 and the firmware streams at 115200. Keep the stream small: VAS1 is 30 Hz with 3-decimal values.
- Opening the serial port must not reset the ESP32 (that would lose its calibration). The server opens with hupcl off and DTR/RTS low.
- The user's main Wi-Fi was an open network, which may block device-to-device traffic. The USB link to the beacon avoids that. The phone still has to reach the laptop on :8080 (app) or :8443 (web).
- The user does **not** want to use a phone hotspot, and can't type on the phone (hence the QR code on the laptop and the auto-calibration).
- Keep the user's words in mind: they want it to "just work". Verify on hardware where you can, report honestly, and avoid long explanations.

## Testing tips

- Raw serial from Python (stop the server first): see how flash.sh finds the port, open at 115200, send `imu\n` to toggle the debug stream.
- WebSocket smoke test: `node -e` with the `ws` client against `ws://localhost:8080/ws` and count `bstate` messages (~30/s).
- Web-page JS can be syntax-checked with `node --check public/headset.js`.
