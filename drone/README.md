# dronekit – open client for Sky Rider Cruise / cooingdv Wi-Fi drones

Standalone Python tool inside VisionAiry. It doesn't use the Node server; run it from this folder (`cd drone`).

- `PROTOCOL.md` – full protocol spec (both hardware variants)
- `protocol.py` – packet builders/parsers + Variant-B video reassembler
- `jpeg_header.py` – regenerates the JPEG headers Variant-B frames need
- `drone.py` – test client (video + keyboard flight)
- `test_sim.py` – self-test against simulated drones (no hardware needed)

```
pip install opencv-python numpy pillow
# join the drone's Wi-Fi, then:
python drone.py                  # video window + flight: W/S A/D move, UP/DOWN height,
                                 #   T take off, L land, SPACE motors off, ESC quit
python drone.py probe            # video + incoming messages, sends NO flight commands
python test_sim.py               # offline self-test
```
