# dronekit – open client for Sky Rider Cruise / cooingdv Wi-Fi drones

Standalone Python tool inside VisionAiry. It doesn't use the Node server; run it from this folder (`cd drone`).

- `PROTOCOL.md` – full protocol spec (both hardware variants)
- `protocol.py` – packet builders/parsers + Variant-B video reassembler
- `jpeg_header.py` – regenerates the JPEG headers Variant-B frames need
- `drone.py` – test client (video + keyboard flight)
- `drift.py` – drift detection and auto-trim from the video
- `target.py` – click-to-go tracking and steering
- `obstacles.py` – finds, names and tracks objects / obstacles in the video
- `detector.py` – the object-naming model (YOLOX via OpenCV DNN, downloaded once into `models/`)
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

### Objects and obstacles

Only these are named on screen, each with an ID number such as `HUMAN #3`, `CHAIR #7` or `DOOR #2`:

| Name | Found by |
| --- | --- |
| human, chair, table, tv, bag (backpack / handbag / suitcase) | a small YOLOX model run by OpenCV (reliable) |
| box | shape rule: a solid outline with 4–6 corners |
| window | shape rule: a bright, roughly rectangular area (daylight windows; not at night, and a lamp or white sheet can fool it) |
| door | shape rule: two tall upright lines a door's width apart with a crossbar on top, reaching toward the floor (a wardrobe can fool it) |
| wall | shape rule: a big blank area covering the middle of the view |

The model doesn't know boxes, windows, doors or walls, so those four are guesses from shapes and brightness and are less reliable than the model's names. Anything else is still tracked, but only appears while it's in your way, as `OBSTACLE`. A door never triggers the warning, since it's the way through.

The model is downloaded once (about 20 MB) into `models/` the first time you run it. Until then, or with no internet, only box / window / door / wall are named. It runs on its own thread, so a slow PC makes the names update less often but never freezes the video. The top bar shows which model is running and how long it takes per frame.

| Flag | Model |
| --- | --- |
| `--model tiny` (default) | ~20 MB, ~50 ms per frame, better on small or far things |
| `--model nano` | ~4 MB, ~25 ms per frame, for slower PCs |
| `--model off` | no names, unlabelled boxes only |

Box colours:

| Colour | Meaning |
| --- | --- |
| green | tracked, named object |
| yellow | getting bigger, so you're closing in on it (shows seconds to contact) |
| red | in the flight path and either very close or under 2 s away: **OBSTACLE AHEAD** blinks |
| grey | briefly lost, still being predicted |

**Built for a moving drone.** Every frame it measures how the whole picture moved (the drone sliding, turning or flying forward) and moves every box with it, so boxes stay on their objects through fast moves. The model's late answers are carried forward the same way. Frames smeared by a sudden jerk are skipped. A name only appears after a few sightings and doesn't flip once it's settled, and a named object keeps its box and ID for up to 1.5 s if it's briefly missed (behind something, or blurred).

It only warns and never steers the drone. A blank wall filling the whole view isn't an "object", so watch the drift panel's zoom for that case. Turn it off with `O` or start with `--no-objects`. To try it without the drone, run `python obstacles.py` (webcam) or `python obstacles.py video.mp4 nano`.

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
| `O` | Object tracking on / off |
| `ESC` | Quit (closing the window also quits) |

Photos and recordings are saved to `media/`. In `probe` mode only the camera and display keys work; the flight and trim keys do nothing.
