# dronekit – open client for Sky Rider Cruise / cooingdv Wi-Fi drones

A standalone Python tool inside VisionAiry. It flies a cheap Wi-Fi toy drone from your laptop instead of the phone app, and adds computer vision on top of the video. It doesn't use the Node server; run everything from this folder (`cd drone`).

## What it does

- **Live video** from the drone in a window with a heads-up display: connection status, Wi-Fi signal, latency, fps, flight timer, stick positions and drift.
- **Keyboard flight**: take off, land, emergency stop, move, climb and descend, with three speed modes like the phone app.
- **Auto-trim**: watches the video to see which way the drone drifts on its own and trims it out. Trims are remembered between flights.
- **Click to go**: click a spot in the video and the drone flies there at the same height.
- **Object identification**: boxes, names and tracks **humans, chairs, tables, TVs, bags, boxes and walls** (windows and doors count as walls), each with an ID (`CHAIR #7`). It's built to stay steady while the drone moves.
- **Obstacle warning**: anything in the flight path that is very close, or that you'll reach in under 2 s, turns red and **OBSTACLE AHEAD** blinks. It only warns; it never steers the drone.
- **Photos and recording** to `media/`.

It works with both hardware versions of these drones (see [Which drone do I have?](#which-drone-do-i-have)) and finds out which one it's talking to by itself.

What it doesn't do:
- **No turning (yaw).** The keyboard has no keys for it, so the drone always faces the same way.
- **No automatic obstacle avoidance.** It only warns you.
- **No Bluetooth.** The drone only has Wi-Fi.

## Requirements

| What | Notes |
| --- | --- |
| Python 3 | tested with 3.14 |
| `opencv-python` | video window, image processing, and runs the object model. Tested with OpenCV 5.0. |
| `numpy` | |
| `pillow` | only for `test_sim.py` |
| Windows (recommended) | Instant key press and release, and the Wi-Fi signal readout, use Windows APIs. On macOS or Linux it still works, but held keys rely on keyboard auto-repeat and there's no signal readout. |
| A laptop with Wi-Fi | it joins the drone's own Wi-Fi network |

```
pip install opencv-python numpy pillow
```

**Object model (about 20 MB):** it's downloaded automatically into `models/` the first time you run it. **The drone's Wi-Fi has no internet**, so do the first run while you're still on your normal internet, for example `python obstacles.py`. Without the model everything still works, but only box and wall get names.

## Quick start

1. `cd drone` and `pip install opencv-python numpy pillow`.
2. While still on normal internet, run `python obstacles.py` once so the object model downloads. Close it with `ESC`.
3. Turn the drone on and join its Wi-Fi from your laptop. The name usually starts with `HD-720P-`, `HD-FPV-`, `HD720-` or `FHD-`.
4. Run `python drone.py`.
5. Click the video window so it has focus, place the drone on a flat surface, press `C` to calibrate the gyro, then `T` to take off.
6. Fly with `W A S D` and the up/down arrows. Press `L` to land, or `SPACE` to cut the motors in an emergency.

## Commands

| Command | What it does |
| --- | --- |
| `python drone.py` | Video window and full flight control (same as `python drone.py fly`) |
| `python drone.py probe` | Video, HUD and object tracking, but sends **no flight commands**. Safe for testing the connection with the drone on the table. |
| `python obstacles.py` | Object identification on your **webcam**, no drone needed |
| `python obstacles.py 1` | Same, on webcam number 1 |
| `python obstacles.py video.mp4` | Same, on a video file |
| `python obstacles.py 0 nano` | Webcam with the faster, smaller model (`tiny`, `nano` or `off`) |
| `python test_sim.py` | Offline self-test against simulated drones. Checks the control packets, video reassembly and both hardware variants. Prints `ALL TESTS PASSED`. |

### Options for `drone.py`

They can be combined, e.g. `python drone.py --camera down --model nano`.

| Option | What it does |
| --- | --- |
| `fly` / `probe` | Mode. `fly` is the default; `probe` sends no flight commands. |
| `--variant a` / `--variant b` | Force the hardware variant instead of detecting it from your IP address |
| `--camera forward` / `--camera down` | Which way the drone's camera points (default `forward`). Changes how drift and click-to-go read the video. Obstacle warnings only work with `forward`. |
| `--save DIR` | Also save every video frame as a JPEG into `DIR` |
| `--no-autotrim` | Start with auto-trim off (`K` toggles it) |
| `--no-objects` | Start with object tracking off (`O` toggles it) |
| `--model tiny` | Object model: about 20 MB, about 50 ms per frame, better on small or far things (default) |
| `--model nano` | Smaller, faster model (about 4 MB, about 25 ms per frame) for slower laptops |
| `--model off` | No model. Only box and wall are named. |
| `-v`, `--verbose` | Print every packet sent to the drone |
| `-h`, `--help` | List all options |

## Controls

Click the video window first so it has focus. Keys typed anywhere else are ignored, and the HUD says **CLICK THIS WINDOW TO CONTROL** when it doesn't have focus.

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
| `O` | Object tracking on / off |
| `H` | Show / hide the on-screen key map |
| `ESC` | Quit (closing the window also quits) |

Photos (`photo_*.jpg`) and recordings (`video_*.avi`) are saved to `media/`. In `probe` mode only these keys work; the flight and trim keys do nothing.

## Object identification

Only these are named on screen, each with an ID number such as `HUMAN #3`, `CHAIR #7` or `WALL #2`:

| Name | Found by |
| --- | --- |
| human, chair, table, tv, bag (backpack / handbag / suitcase) | a small YOLOX model run by OpenCV (reliable) |
| box | shape rule: a solid outline with 4–6 corners |
| wall | shape rules, any of: a big blank area covering the middle of the view; a window (a bright, roughly rectangular area); a door (two tall upright lines a door's width apart with a crossbar on top, reaching toward the floor) |

Windows and doors are shown as `WALL`, so in the flight path they set off the obstacle warning like any wall, an open door included. The model doesn't know boxes or walls, so those are guesses from shapes and brightness and are less reliable than the model's names. Anything else is still tracked, but only appears while it's in your way, as `OBSTACLE`.

The model runs on its own thread, so a slow laptop makes names update less often but never freezes the video. The top bar shows the model and its time per frame (e.g. `TINY 50 ms`), `MODEL LOADING` while it starts, or `NO MODEL` if it couldn't be downloaded.

**Built for a moving drone.**
- Every frame it measures how the whole picture moved (the drone sliding, turning or flying forward) and moves every box with it, so boxes stay on their objects through fast moves. The model's late answers are carried forward the same way.
- Frames smeared by a sudden jerk are skipped.
- A name only appears after a few sightings and doesn't flip once it's settled.
- A named object keeps its box and ID for up to 1.5 s if it's briefly missed (behind something, or blurred).

Each kind of object has its own colour, and a key in the top-left lists the kinds on screen right now:

| Colour | Object |
| --- | --- |
| magenta | human |
| sky blue | chair |
| green | table |
| purple | tv |
| orange | bag |
| tan | box |
| light grey | wall (including windows and doors) |

Red and yellow are kept for warnings:

| Look | Meaning |
| --- | --- |
| yellow seconds after the name (e.g. `CHAIR #4  3.2s`) | getting bigger, so you're closing in on it; that's the time to contact |
| thick red brackets | in the flight path and either very close or under 2 s away: **OBSTACLE AHEAD** blinks. Unnamed things in the way show as red `OBSTACLE`. |
| thin, dim brackets | briefly lost, still being predicted |

Limits:
- Small or far-away things and poor light make names less reliable.
- In a blank or dark scene the camera motion can't be measured, so boxes follow less well.
- Something that leaves the view and comes back gets a new ID.

## The screen

| Where | What |
| --- | --- |
| Top bar, left | `LIVE` / `CONNECTING` / `SIGNAL LOST`, hardware variant, drone Wi-Fi name and signal |
| Top bar, right | Video delay in ms, resolution, fps, number of objects, object model and its speed, flight timer, recording timer |
| Top-left | Speed buttons 1 / 2 / 3 |
| Bottom-left | Key map (`H` hides it) |
| Bottom-right | Stick positions (X/Y pad and Z bar) with the trim offset |
| Right, above that | Drift: which way the drone is sliding and what auto-trim is doing (`CALIBRATING`, `CORRECTING`, `HOLDING`, `PAUSED`) |
| Centre | Crosshair, object boxes and names, the `GO TO` ring, and banners (`TAKING OFF`, `LANDING`, `OBSTACLE AHEAD`, messages) |

## Which drone do I have?

There are two hardware variants. `drone.py` works out which one from the IP address your laptop gets on the drone's Wi-Fi:

| Your IP | Variant | Video | Control |
| --- | --- | --- | --- |
| 192.168.1.x | **A** | RTSP `rtsp://192.168.1.1:7070/webcam` | UDP to 192.168.1.1:7099 |
| 192.168.169.x | **B** | custom UDP on 192.168.169.1:8800 | same socket |

If it says *Not on a drone network*, you aren't connected to the drone's Wi-Fi yet. Join it, or force a variant with `--variant`. The full protocol is in [PROTOCOL.md](PROTOCOL.md).

## Files

| File | What it is |
| --- | --- |
| `drone.py` | The main program: connection, keyboard flight, video window |
| `hud.py` | Draws the heads-up display |
| `protocol.py` | Packet builders and parsers, Variant-B video reassembler |
| `jpeg_header.py` | Regenerates the JPEG headers Variant-B frames need |
| `drift.py` | Drift detection and auto-trim from the video |
| `target.py` | Click-to-go tracking and steering |
| `obstacles.py` | Finds, names and tracks objects; camera-motion tracking; obstacle warnings. Runs on its own on a webcam or video. |
| `detector.py` | The object-naming model (YOLOX via OpenCV DNN, downloaded once into `models/`) |
| `test_sim.py` | Offline self-test against simulated drones |
| `PROTOCOL.md` | Full protocol spec for both hardware variants |
| `trim.json` | Saved trims (written automatically) |
| `models/` | Downloaded object models (not in git) |
| `media/` | Your photos and recordings (not in git) |
