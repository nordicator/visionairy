# XRAY: see motion through walls with Wi-Fi

This is a simple recreation of Hack Canada's *Wallhacks*. The original used mmWave radar. This version uses only an ESP32 and a phone hotspot.

- **Transmitter:** your iPhone Personal Hotspot.
- **Receiver:** a Freenove ESP32-WROOM. It joins the hotspot and pings it about 100 times a second. For each reply it reads the Wi-Fi **Channel State Information (CSI)**, which is the amplitude of 51 OFDM subcarriers.
- **Detection (runs on the ESP32):** each frame is normalised by its mean so gain changes don't count as motion. "Turbulence" is the average per-subcarrier standard deviation over a 0.5 s window. The first 8 s after connecting set the empty-room baseline. Score = turbulence ÷ baseline, smoothed, with hysteresis. When the score passes the threshold, the onboard LED (GPIO2) lights and a `CONTACT` is reported.
- **Visualizer (laptop):** `server.js` reads the ESP32 over USB and shows a radar scope, the score history and a live CSI waterfall at http://localhost:8090.

A single link detects *that* someone is moving through the space between the phone and the ESP32, not *where* they are.

## Setup

1. iPhone: **Settings → Personal Hotspot**. Turn on **Allow Others to Join** and **Maximize Compatibility** (forces 2.4 GHz). Keep that screen open until the ESP32 connects.
2. Put your hotspot credentials in `xray/.env` (see `.env.example`).
3. Flash the board, then start the visualizer:

```
cd xray
npm install
./flash.sh          # needs arduino-cli with the esp32:esp32 core
npm start           # open http://localhost:8090
```

Stop `npm start` before re-flashing, because both need the serial port.

## Placement

Put the phone on one side of the wall and the ESP32 on the other, ideally 2–5 m apart and at chest height. Leave the area empty during the 8 s calibration (LED blinking), then walk through the line between them. Press **C** to recalibrate after moving anything. Raise the threshold if it triggers on nothing; lower it if it misses you.
