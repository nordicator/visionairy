# VisionAiry

DIY F-35 style helmet that shows where an ESP32 beacon is, live in AR and in a 3D graph, tracked by IMUs only.

## How it works

No camera tracking. Everything is coordinates from motion sensors.

1. **Calibrate:** lay the beacon flat and the phone face up on top of it, with the phone's top edge along the IMU's printed **Y arrow**. After a 2-second countdown it calibrates by itself. The beacon learns its gyro bias, accelerometer scale and which way is up. The phone records its heading. That spot becomes the **origin** for both: X right, Y along the arrow, Z up, in metres.
2. **Track:** the beacon dead-reckons its position (gyro → orientation, accelerometer → velocity → position) at 500 Hz and streams it 30×/s. The phone does the same with its own motion sensors. Two physical facts keep the drift down:
   - **Zero-velocity updates:** whenever a device is still (beacon: 80 ms, phone: 250 ms), its velocity is exactly zero.
   - **Free fall:** once the beacon is airborne, the only force is gravity, so flight uses exact ballistics instead of the sensor.
3. **See it:** the phone shows the beacon in AR as a live dot with a trailing line (yellow while flying), its coordinates and its distance. **`/graph`** on the laptop shows both devices in a rotatable 3D view: grid, axes, beacon board, phone camera and both trails.

**Drift:** position from integrated accelerometers wanders while things are moving (expect centimetres to tens of centimetres after a few seconds of motion; nothing while still). To reset it, **touch the beacon to the phone and hold the screen**. That re-zeroes both to the same point.

## iPhone AR app (recommended)

`ios/` is a native app that uses **ARKit** to track the phone: millimetre-level position from the camera and motion sensors, the exact lens, and detected floors, tables and walls (drawn faintly in blue). That's what makes the overlay line up. A web page can't do this on an iPhone. The beacon itself is still tracked only by its IMU.

```
cd ios
./build.sh          # builds, installs on the plugged-in iPhone and launches it
```

The laptop's current Wi-Fi address is baked into the app as the server, so re-run `build.sh` if it changes. The first time, enable **Developer Mode** on the iPhone (Settings → Privacy & Security) and trust the developer (Settings → General → VPN & Device Management).

**Calibrate:** hold the phone flat, screen up, at waist height, and lay the beacon on the screen with its Y arrow toward the top of the phone. Hold still for 2 seconds. The camera keeps seeing the floor, so ARKit keeps tracking. **Re-zero:** touch the beacon to the phone and tap RE-ZERO.

## Run

```
npm install
npm start
```

Open the printed `https://<laptop-ip>:8443/headset` on the headset phone and accept the certificate warning. The laptop, the phone and the ESP32 must be on the same Wi-Fi (e.g. the phone's hotspot). On the laptop, `https://localhost:8443/headset` with split-screen unticked works too; press **T** for a fake throw to preview the rendering.

## Beacon

```
cd esp32
cp .env.example .env    # hotspot name + password
./flash.sh
```

The board needs power in flight: use a small USB power bank taped to it. Its LED blinks until calibrated, then stays solid. An optional external LED on **GPIO26** does the same.

**IMU (MPU-6050 / 6500):** VCC → 3V3, GND → GND, SDA → **GPIO32**, SCL → **GPIO33**. Fix it firmly to the board.

## Using it

1. Open `https://localhost:8443` on the laptop and scan the QR code with the phone. On the phone, press ENGAGE. On the laptop, open **3D GRAPH**.
2. Lay the phone face up on the beacon (top edge along the Y arrow) and wait for **CALIBRATED // ORIGIN SET**.
3. Pick both up and move the beacon around or throw it. Watch the dot and trail on the phone and in the graph.
4. Tap to clear the trail. To re-zero, touch the beacon to the phone and hold the screen.

Laptop keys on /headset: **K** calibrate, **Z** re-zero, **C** clear.
