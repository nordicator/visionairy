# dronekit – open client for Sky Rider Cruise / cooingdv Wi-Fi drones

Standalone Python tool inside VisionAiry. It doesn't use the Node server; run it from this folder (`cd drone`).

- `PROTOCOL.md` – full protocol spec (both hardware variants)
- `protocol.py` – packet builders/parsers + Variant-B video reassembler
- `jpeg_header.py` – regenerates the JPEG headers Variant-B frames need
- `drone.py` – test client (video + keyboard flight)
- `drift.py` – drift detection and auto-trim from the video
- `target.py` – click-to-go tracking and steering
- `test_sim.py` – self-test against simulated drones (no hardware needed)

```
pip install opencv-python numpy pillow
# join the drone's Wi-Fi, then:
python drone.py                  # video window + flight (same as `python drone.py fly`)
python drone.py probe            # video + incoming messages, sends NO flight commands
python drone.py --save frames/   # also save every video frame as JPEG
python test_sim.py               # offline self-test
```

## Controls

Click the video window first so it has focus; keys typed anywhere else are ignored.

### Flight

Hold a movement key to push harder; let go and that stick recenters.

| Key | Action |
| --- | --- |
| `W` / `S` | Forward / back |
| `A` / `D` | Left / right |
| `UP` / `DOWN` (or `R` / `F`) | Climb / descend |
| `T` | Take off |
| `L` | Land |
| `SPACE` | Motors off (emergency stop) |
| `C` | Gyro calibrate (only on the ground) |

### Speed

Click the **SPEED 1 / 2 / 3** buttons in the top-left of the window; there are no keyboard keys for them. They work like the app's speed modes: they limit how far the left/right and forward/back stick goes when you hold a key (speed 1 = ±50, 2 = ±60, 3 = ±127 out of 127). Up/down always uses the full range, and so do trim and click-to-go. It starts on speed 1, the app's default.

### Click to go

Once the drone is in the air, click anywhere on the video and it flies towards that spot. A pulsing **GO TO** ring marks the target. **The height never changes**: the throttle stays on hold-altitude, so only X (left/right) and Y (forward/back) are used.

| Camera | What it does |
| --- | --- |
| forward (default) | Flies left/right until the spot is centred, then forward until the spot looks twice as big as when you clicked it |
| down (`--camera down`) | Flies left/right and forward/back until the spot is under the crosshair |

The spot is tracked in the video while the drone moves. It stops when it gets there, loses sight of the spot, or after 20 seconds. Because the height is held, a spot well above or below the drone slides off the top or bottom of the forward camera as it gets closer, and the drone stops there. **Right-click** or any movement key cancels it, and so do landing and motors off. It works best on textured objects a few metres away. The first click on an unfocused window only focuses it. Click-to-go uses its own speed limits, whatever speed you've picked.

| Mouse | Action |
| --- | --- |
| Left-click on video | Fly to that spot |
| Right-click | Cancel |

### Trim (drift correction)

| Key | Action |
| --- | --- |
| `SHIFT` + `W` / `S` / `A` / `D` / `R` / `F` | Nudge the trim 2% in that direction |
| `K` | Turn auto-trim on or off |
| `BACKSPACE` | Reset trim to zero |

Auto-trim watches the video to see which way the drone is sliding and adjusts the trim to cancel the drift. Each time you stop moving (0.7 s after you release the keys), and again right after take off, it calibrates for 3 seconds with faster correction (the HUD shows **CALIBRATING**), then goes back to a slow hold. Pressing a movement key interrupts the calibration. Trims are saved to `trim.json` and reused next time.

### Camera and display

| Key | Action |
| --- | --- |
| `P` | Save a photo |
| `V` | Start / stop recording |
| `H` | Show / hide the on-screen key map |
| `ESC` | Quit (closing the window also quits) |

Photos and recordings are saved to `media/`. In `probe` mode only the camera and display keys work; the flight and trim keys do nothing.
