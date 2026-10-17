# VisionAiry

DIY F-35 style "distributed aperture" helmet. Cameras around the room plus a phone in a cardboard headset let you look around and see through your own hands.

## Run

```
npm install
npm start
```

On the laptop running the server, open **`https://localhost:8443/control`**. It shows QR codes for the other devices, a top-down coverage map, a live thumbnail of every camera, and sliders to retune any camera remotely. All devices must be on the same Wi-Fi; accept the certificate warning on each.

- **Room cameras** → scan the ROOM CAMERA code, tap a **position preset**, then START.
- **Headset** → scan the HEADSET code, tap ENGAGE, put the phone in the box, look at any room camera's view, and **hold** to sync.

## Our layout (2 phones + 3 laptops + headset phone)

Every camera sits ~0.8 m from the pilot (just outside arm's reach) at chest height, lens pointing **away** from the pilot. Laptops face their screen outward.

| Device | Preset | Heading / pitch |
|--------|--------|-----------------|
| Phone A | FRONT, tilted slightly down so it covers eye level and your hands (the XRAY camera) | 0° / -15° |
| Phone B | BELOW, taped under your seat looking straight down, top edge of the frame toward the front | 0° / -90° |
| Laptop 1 | RIGHT | 90° / 0° |
| Laptop 2 | BEHIND | 180° / 0° |
| Laptop 3 | LEFT | 270° / 0° |

Laptops: plug in, disable sleep (`caffeinate -d`), keep the camera tab in front. Phones: landscape, level, on chargers.

## Test plan

1. **Dry run, one laptop:** open `/control`, press ADD TEST CAMERAS, open `/headset` in another window, press 2 (DAS) and drag around. The labeled angle grids should meet across seams.
2. **Placement:** set up the real devices; every ring on the `/control` map that should be covered turns green. Red = gap.
3. **Tuning:** one person wears the headset in XRAY and holds a hand toward Phone A. Someone at `/control` drags that camera's FOV until the room behind the hand is the same size as around it, then nudges heading/pitch until it lines up. Repeat in DAS for the laptops.

## Headset modes (tap to cycle)

| Mode | What you see |
|------|--------------|
| XRAY | Headset camera passthrough. Your hands are cut out and filled with the room camera facing that way. With no room camera there, it uses a remembered hand-free frame. |
| DAS  | Room cameras only, stitched around you by direction. Turn your head to look through the "jet". |
| PASS | Plain passthrough with HUD. |

## Staying lined up

The phone gyro only knows relative turning and drifts, so the headset lines itself up with the room cameras by matching edges in its own camera view against theirs.

- **Hold** (or R) does a full search across every camera, so you can face any direction. If nothing matches it falls back to "I'm facing heading 0".
- **Auto-sync** (on by default) keeps refining while you look around, which also corrects cameras placed up to ~12° off their preset. `SYNC 0.xx` on the HUD is the match quality; `SYNC --` means no recent match (blank walls, looking at the floor camera, or hands in view).
- It works best on a textured scene. Plain walls give it nothing to grab.

## Tips

- Stay seated: there's no position tracking, so cameras only line up from where the pilot sat when recentering.
- iPhone: turn off rotation lock and hold the phone in landscape. For true fullscreen, use Share → Add to Home Screen.
- Hand tracking downloads a MediaPipe model from the internet the first time it runs.
- Laptop testing: drag to look, keys 1/2/3 switch modes, R syncs.
