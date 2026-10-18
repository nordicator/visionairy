# dronekit – open client for Sky Rider Cruise / cooingdv Wi-Fi drones

A standalone Python tool inside VisionAiry. It flies a cheap Wi-Fi toy drone from your laptop instead of the phone app, and adds computer vision on top of the video. It doesn't use the Node server; run everything from this folder (`cd drone`).

## What it does

- **Live video** from the drone in a window with a heads-up display: connection status, Wi-Fi signal, latency, fps, flight timer, stick positions and drift.
- **Keyboard flight**: take off, land, emergency stop, move, climb and descend, with three speed modes and headless mode like the phone app.
- **Auto-trim**: watches the video to see which way the drone drifts on its own and trims it out. Trims are remembered between flights.
- **Click to go**: click a spot in the video and the drone flies there at the same height.
- **Object identification**: boxes, names and tracks **humans, chairs, tables, TVs, bags, boxes and walls** (windows and doors count as walls), each with an ID (`CHAIR #7`). It's built to stay steady while the drone moves.
- **Distance (depth)**: how far away every object is in metres (`CHAIR #7  1.8 m`), plus its 3D position and direction, and how far it is to whatever is straight ahead. This is the base for flying to objects by voice later.
- **Obstacle warning**: anything in the flight path that is very close, or that you'll reach in under 2 s, turns red and **OBSTACLE AHEAD** blinks. It only warns; it never steers the drone.
- **AI pilot** (`--ai`): type what you want (`take off, fly a 1 m square, land`) and an LLM flies it, with an obstacle guard, position hold and self-correction underneath. See [AI pilot](#ai-pilot).
- **Photos and recording** to `media/`.

It works with both hardware versions of these drones (see [Which drone do I have?](#which-drone-do-i-have)) and finds out which one it's talking to by itself.

What it doesn't do:
- **No turning (yaw) from the keyboard.** There are no keys for it; only the AI pilot turns.
- **No obstacle avoidance when you fly by hand.** It only warns you; the AI pilot's guard stops its own plans.
- **No Bluetooth.** The drone only has Wi-Fi.

## Requirements

| What | Notes |
| --- | --- |
| Python 3 | tested with 3.14 |
| `opencv-python` | video window, image processing, and runs the object model. Tested with OpenCV 5.0. |
| `numpy` | |
| `onnxruntime-directml` | runs the depth model, on the GPU when there is one (AMD, Intel or NVIDIA on Windows). Use plain `onnxruntime` on macOS or Linux. |
| `pillow` | only for `test_sim.py` |
| Windows (recommended) | Instant key press and release, and the Wi-Fi signal readout, use Windows APIs. On macOS or Linux it still works, but held keys rely on keyboard auto-repeat and there's no signal readout. |
| A laptop with Wi-Fi | it joins the drone's own Wi-Fi network |

```
pip install opencv-python numpy pillow onnxruntime-directml
```

**Object model (about 20 MB):** it's downloaded automatically into `models/` the first time you run it. **The drone's Wi-Fi has no internet**, so do the first run while you're still on your normal internet, for example `python obstacles.py`. Without the model everything still works, but only box and wall get names.

## Quick start

1. `cd drone` and `pip install opencv-python numpy pillow onnxruntime-directml`.
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
| `python drone.py --webcam` | **Test window on the laptop's webcam**, no drone needed: the same screen with object boxes, distances, depth view (`Z`) and depth calibration (`M`). Sends nothing anywhere. `--webcam 1` picks another camera. |
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
| `--depth fast` | Distance model at 420x252: about 50 ms per frame on a GPU (default) |
| `--depth fine` | Sharper distance model at 630x378: about 430 ms per frame, so updates are slower |
| `--depth off` | No distances |
| `--webcam N` | Use webcam N (default 0) instead of the drone; no flight commands. Its lens and depth calibration are kept separately (`camera_webcam.json`, `depth_cal_webcam.json`; for the lens use `calibrate_camera.py --webcam`). |
| `--hfov DEG` | Camera field of view used for distances until the lens is calibrated (default 90 for the drone, 65 for a webcam) |
| `--cal-distance M` | For depth calibration (key `M`): how many metres the wall is from the camera (default 1) |
| `-v`, `--verbose` | Print every packet sent to the drone |
| `-h`, `--help` | List all options |

## AI pilot

Type what you want in the terminal (`take off, fly a 1 m square, turn 90° right, land`) and an LLM flies it through tool calls. Works with any OpenAI-compatible API, including a Spar IC key.

```
cp .env.example .env              # put in SPAR_API_KEY and SPAR_BASE_URL
python ai_pilot.py --list-models  # pick one that supports tool calling, set SPAR_MODEL
python ai_pilot.py --dry-run "take off, forward 1 s, land"   # no drone: prints the exact stick bytes
python drone.py --ai              # real flight: video window + an ai> prompt here
```

The LLM sets each stick (`right`, `forward`, `up`, `turn`) as a % of its range for an exact time, with optional ramps, and flies a whole command (take off, manoeuvres, landing; up to 45 s) as one `sequence`. The LLM thinks once on the ground, so the drone never hovers in the air drifting while it waits. Ramps never change how far a step goes, and the radio thread reads the plan when each packet is sent, so step timing doesn't depend on the video window keeping up. It can also take off, land, cut the motors, fly to a tracked object, set trim, read the status (drift, objects, obstacle warnings) and look at the camera (the frame has the tracked objects outlined with their ids; `--no-ai-vision` turns this off). Unlike the keyboard, it can also yaw.

Reasoning models are asked for low effort by default (`SPAR_REASONING_EFFORT`; `none` turns thinking off). Gemini 3.8 Flash (free tier, base URL `https://generativelanguage.googleapis.com/v1beta/openai`) with `SPAR_REASONING_EFFORT=none` answers in ~3 s and flew every simulator test correctly. `SPAR_MODEL` can list fallbacks (`spur-glm-5-3,spur-qwen3-235b`): if one stalls for 30 s (`SPAR_TIMEOUT`) or errors, the next answers.

Speeds aren't measured, so distances are estimates. Write your own measurements in `ai_notes.md` (e.g. `forward 40% for 1 s = 1.2 m`) and they're added to the LLM's instructions.

### First flight with a new drone (`preflight.py`)

```
python preflight.py              # checks, nothing flies: drone link, video smoothness, camera texture,
                                 #   internet + LLM tool calling while on the drone's Wi-Fi, saved trim
python preflight.py --fly        # calibration hop (flat floor, ~1.5 m clear): takes off, nudges right /
                                 #   forward / turn and back, asks you what you saw, lands
python drone.py --ai --first-flight   # gentle first AI flight: sticks 40%, safe area 1.5 m, ceiling 1.6 m, 3 min
```

**Internet while on the drone's Wi-Fi.** The drone's network has no internet, so the LLM can't be reached unless the laptop has a second connection: Ethernet or USB-tethering a phone, placed above Wi-Fi in System Settings > Network > Set Service Order. `preflight.py` tells you if this is missing.

The hop compares what you saw with what the camera measured, which separates the two things that can be reversed on a new drone: the stick direction (saved as `invert`, fixed in the sticks for keyboard and AI) and the camera's slide estimate (saved as `estimate_flip`, fixed before auto-trim, hold, the HUD and the AI read it). It also checks the altitude hold, estimates the video lag, and if you give rough distances writes the real speeds into `ai_notes.md`. Everything goes in `calibration.json`, which `drone.py` applies on every start (`--no-calibration` to ignore it). Rehearse it on the simulator with `python preflight.py --sim --fly` (add `--invert x --sign-bug --latency 0.3` for a misbehaving drone).

### Live safety and self-correction (`supervisor.py`)

The LLM takes seconds to answer, so a faster layer runs every video frame underneath it and fixes what it can by itself. Every action shows on the HUD, goes in the flight log, and is reported to the LLM (`safety_alerts`, `auto_landing` in its status).

| What goes wrong in the room | What it does about it |
| --- | --- |
| Draft / fan / bad trim makes it slide | Position hold pushes back (12-20% stick) and builds up an integral term, so a steady draft ends up cancelled. Holds from 1 s after take off, in hover steps, while the LLM thinks, and while landing |
| A strong air current it can't beat at 20% | Raises its push limit 5% at a time up to 35%; if even that can't stop the slide, it lands |
| A sudden shove | Brakes and comes back |
| The camera's slide estimate is reversed on this drone | Notices the slide growing while it pushes against it, takes back the trim, stops auto-trim on that axis and flips the axis; if flipped is wrong too, switches that axis off (and lands if it then drifts hard) |
| Hold settings too aggressive for this drone (wobbling) | Notices the slide reversing back and forth and softens its own gain and minimum push |
| The drone turns by itself (gyro bias) | Calibrates the gyro automatically before the first AI take off of the session (`--no-ai-gyro` to skip). In the air this can't be told apart from a draft with one camera, so it only warns (`HEAVY X CORRECTION`) when a large correction is held for long |
| Something ahead while flying forward | Obstacle guard stops the plan; the LLM gets the reason, the objects and a labelled frame, and replans |
| A plain wall the object tracker can't see | Stops a forward step when the camera loses all texture (blank wall close, or too dark) |
| Drifting toward something while hovering | Backs away from anything very close or closing in ahead |
| Video lost | Stops the plan at once; lands after 5 s without video (`--ai-lost-land`) |
| Battery | Warns at 5 min of flight, lands at 7 min (`--ai-max-flight`) |
| The LLM plans something too big for a room | Every plan is dead-reckoned first; ones that go more than 3 m from the take-off spot (`--ai-max-distance`), above 2 m (`--ai-ceiling`) or below 0.3 m without landing are refused with the reason |
| You quit (ESC, closing the window, Ctrl-C) or the program crashes mid-air | It sends a landing command and keeps the radio running 3.5 s before exiting |

| Manual safety | |
| --- | --- |
| `--ai-max-stick` | stick cap, default 60% |
| any movement key, T, L, SPACE, video lost | stops the AI's manoeuvre and its movement for that command |
| `stop` / `land` / `kill` typed in the terminal | act at once, without the LLM |
| a new command | cancels the current one |
| `--no-ai-guard`, `--no-ai-hold` | turn the guard / position hold off |
| `--ai-hold-min`, `--ai-hold-gain`, `--ai-hold-max` | hold tuning (it also tunes itself, see above) |

`python test_faults.py` injects each of these problems into the simulator and checks the result (18 scenarios including 0.4-0.5 s of video lag, ~40 s).

### AI panel (top left of the video window)

| Line | Shows |
| --- | --- |
| `AI <model>` | which model answered last, and IDLE / THINKING 2.3s / RUNNING <tool> |
| `> ...` | the command you typed |
| `PLAN step 4/9: forward +30  1.2s left` | the step being flown; the strip under it is the whole plan (amber = move, grey = hover, green = take off, yellow = land, white line = now) |
| `PLAN DONE` / `PLAN ABORTED: ...` (red) | how the last plan ended, for 8 s |
| `AI: ...` | the LLM's last reply |
| `GUARD WATCHING / STANDBY / STOPPED: why / OFF` | the obstacle guard (red when it stopped a plan or is blind) |
| `HOLD ON X-6 Y+4  max 25/20%` / `HOLD WAITING: reason` | the position hold's push (and its raised limits), or why it's not holding; `(X OFF)` if an axis was switched off |
| yellow / red lines under it | safety alerts: strong drift, heavy correction, axis flipped, wobble softened, video lost, battery, backing away, auto-landing (red = it's acting) |
| `AUTO-LANDING` (flashing, centre) | it's landing by itself; the reason is in the panel |
| `log: logs/...` | this session's flight log |

### Flight log

Every run of `drone.py` writes `logs/flight_<time>.jsonl` (simulator and dry runs write `sim_` / `dryrun_` logs): the command, every LLM request and reply (model, seconds, tokens, full tool calls), every tool result, plan steps, guard stops with the object, hold changes, actions, link changes, errors with tracebacks, and 5 Hz telemetry (sticks sent, hold, trim, drift estimate, objects, plan step, keys). Read it with `python flightlog.py` (newest log, telemetry left out), `--tel` to include telemetry, `--tail N` for the end.

### Testing without the drone

```
python test_ai.py      # plan timing, ramps, validation, overrides, guard, hold, HUD panel (fake LLM)
python test_ui.py      # drone.py's real video loop headless: AI plan, guard stop, hold, HUD, flight log, land on quit
python test_faults.py  # 18 room failures injected into the simulator (drafts, gale, shove, reversed camera, lag, ...)
python test_preflight.py  # the calibration hop on simulated drones reversed in different ways
python test_launch.py     # `drone.py --ai` launched for real against a fake drone: checks the bytes it sends, and the landing on ESC
python sim_flight.py "take off, fly forward 3 metres, go around anything in the way, then land" --obstacle 0,1.6
python sim_flight.py "take off, hover in place for 10 seconds, then land" --wind 0.15 [--no-hold] [--no-autotrim]
```

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
| `H` | Headless mode on / off: **forward** means away from you, whichever way the drone's nose points. Point the drone away from you before switching it on. |

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

Auto-trim watches the video to see which way the drone is sliding and adjusts the trim to cancel the drift. Each time you stop moving (0.7 s after you release the keys), and again right after take off, it calibrates for 3 seconds with faster correction (the top bar shows **CALIBRATING DRIFT**), then goes back to a slow hold. Pressing a movement key interrupts the calibration. Trims are saved to `trim.json` and reused next time.

### Camera and display

| Key | Action |
| --- | --- |
| `P` | Save a photo |
| `V` | Start / stop recording |
| `O` | Object tracking on / off |
| `Z` | Depth view on / off: colours the video by distance (red near, blue far) |
| `M` | Depth calibration: see [Distance (depth)](#distance-depth) |
| `?` | Show / hide the list of keys |
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

## Distance (depth)

Each object's name tag shows its distance, e.g. `CHAIR #7  1.8 m`. A `~` (as in `~1.8 m`) means the distance is being predicted from the drone's motion rather than measured just now. Under the crosshair is the distance to the nearest thing straight ahead. **OBSTACLE AHEAD** blinks when that's under 0.8 m, or when an object in the flight path is closer than that.

A single camera can't measure distance directly like two eyes; a model judges it from how things look. It uses [Depth Anything V2 Metric Indoor](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf), which is trained on indoor scenes and gives metres. Four things make those metres trustworthy:

1. **Lens correction.** The model assumes the camera it was trained with, but toy-drone lenses are wider, which makes everything look farther away. Distances are corrected for the drone's actual lens. That lens is guessed (90° wide) until you measure it with a checkerboard (below).
2. **Tape-measure calibration.** Measures what's left of this camera's error.
3. **Steadiness.** Each new depth map is shifted to follow the drone's movement and blended with the last, so it doesn't flicker. In tests, distances changed less than 1% from frame to frame.
4. **Near surface.** An object's distance comes from the closest part of the middle of its box, so it's the front of the chair, not the wall behind it.

**Calibrate once per drone, for the best accuracy:**

1. **Lens.** Print a checkerboard (search "opencv chessboard 9x6") and tape it to something flat. Run `python drone.py probe --save calib/` and show the board to the camera from 15–30 different angles and distances. Then run `python calibrate_camera.py calib/`, which writes `camera.json`. On synthetic tests it measured the lens to within 0.2%.
2. **Scale.** Point the camera straight at a flat wall exactly 1 m away (tape measure) and press **M**. For other distances, start with `--cal-distance 2`. Pressing M at a few different distances averages them. The result is saved in `depth_cal.json`.

**Safety margin:** warnings (and later, flying to objects) assume everything is **20% closer** than the estimate. Getting it wrong in that direction means stopping early; the other way means crashing. The numbers on screen are the plain estimate; code should plan with `dist_safe`. To change the margin, edit `SAFETY` in `obstacles.py`.

Accuracy to expect:
- **Uncalibrated:** roughly ±20–40%.
- **After both steps:** typically ±5–10% indoors, at 0.3–5 m.

It's worst on plain featureless walls and glass, very close up (under about 0.3 m) and in poor light. Objects cut off by the frame edge keep their last measured distance, carried forward by the drone's motion.

**For code (voice commands):** `ObstacleTracker.objects()` gives, for each object:

| Field | Meaning |
| --- | --- |
| `dist` | metres to its near surface |
| `dist_fresh` | `True` if just measured, `False` if predicted |
| `dist_safe` | `dist` with the safety margin taken off: plan with this one |
| `xyz` | metres: right, down, forward |
| `bearing` | degrees right of straight ahead |

`ahead_info()` gives the distance to whatever is straight ahead. [depth.py](depth.py) has the details.

The depth model is about 99 MB, in `models/`. It's built from the official weights with `python export_depth.py`, which needs PyTorch. If it's missing, drone.py downloads a ready-made version instead, which is slower and squashes the picture slightly.

## The screen

| Where | What |
| --- | --- |
| Top bar, left | `LIVE` / `CONNECTING` / `SIGNAL LOST` and the drone's Wi-Fi signal |
| Top bar, middle | Modes, only while they apply: `HEADLESS`, `CALIBRATING DRIFT`, `AUTO-TRIM OFF`, `PROBE` |
| Top bar, right | Video delay, fps, flight time, recording time. Object and depth model status only appear if something is wrong (`MODEL LOADING`, `NO DEPTH`, ...) |
| Centre | Crosshair with the distance to whatever is straight ahead, object boxes with names and distances, the `GO TO` ring, and banners (`TAKING OFF`, `LANDING`, `OBSTACLE AHEAD`, messages) |
| Bottom-left | Speed buttons 1 / 2 / 3 (click), and `? keys` |

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
| `depth.py` | Distance model (Depth Anything V2 Metric Indoor via onnxruntime), lens and tape calibration, depth smoothing |
| `export_depth.py` | Builds the depth model files in `models/` from the official weights (needs PyTorch) |
| `calibrate_camera.py` | Measures the camera's lens from checkerboard photos and writes `camera.json` |
| `test_sim.py` | Offline self-test against simulated drones |
| `PROTOCOL.md` | Full protocol spec for both hardware variants |
| `trim.json` | Saved trims (written automatically) |
| `ai_pilot.py` | The AI pilot: LLM client, plan scheduler, tools, safe-area check (see [AI pilot](#ai-pilot)) |
| `supervisor.py` | Fast safety layer under the AI: obstacle guard, position hold, self-checks, watchdogs; the HUD's AI panel state |
| `calibration.py` | What the calibration hop learned (`calibration.json`), applied to every flight |
| `preflight.py` | Pre-flight checks and the calibration hop |
| `flightlog.py` | The JSON-lines flight log in `logs/`, and a viewer for it |
| `sim_flight.py` | Runs an AI command on a simulated drone and draws the path; fault injection |
| `test_ai.py` | Offline self-test of the AI pilot (fake LLM) |
| `test_ui.py` | drone.py's video loop run headless with the AI wiring |
| `test_launch.py` | `drone.py --ai` launched for real against a fake drone |
| `test_faults.py` | Room failures injected into the simulator; checks the self-corrections |
| `test_preflight.py` | The calibration hop rehearsed on misbehaving simulated drones |
| `models/` | Downloaded object models (not in git) |
| `logs/` | Flight logs (not in git) |
| `media/` | Your photos and recordings (not in git) |
