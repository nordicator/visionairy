# VisionAiry

DIY F-35 style "distributed aperture" helmet. Cameras around the room plus a phone in a cardboard headset let you look around and see through your own hands.

## Run

```
npm install
npm start
```

Open the printed `https://<laptop-ip>:8443` URL on every device and accept the certificate warning. All devices must be on the same Wi-Fi.

- **Room cameras** → `/camera`. Set **heading** (which way the lens points: 0° = pilot's front, 90° = right, 180° = behind) and **pitch** (-90° = floor). Then tap START.
- **Headset** → `/headset`. Tap ENGAGE, put the phone in the box, look straight ahead, and **hold** to recenter.

## Headset modes (tap to cycle)

| Mode | What you see |
|------|--------------|
| XRAY | Headset camera passthrough. Your hands are cut out and filled with the room camera facing that way. With no room camera there, it uses a remembered hand-free frame. |
| DAS  | Room cameras only, stitched around you by direction. Turn your head to look through the "jet". |
| PASS | Plain passthrough with HUD. |

## Tips

- Put the room cameras **near the pilot, pointing outward** (like the cameras on the jet's skin). A phone on the floor pointing down at pitch -90 gives the "look through the floor" shot.
- iPhone: turn off rotation lock and hold the phone in landscape. For true fullscreen, use Share → Add to Home Screen.
- Hand tracking downloads a MediaPipe model from the internet the first time it runs.
- Laptop testing: drag to look, keys 1/2/3 switch modes, R recenters.
