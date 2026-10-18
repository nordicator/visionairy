"""Voice commands for the AI pilot: talk instead of typing at the "ai>" prompt.

  python voice.py          # test: press Enter, say something, see what it heard and how long it took

Press Enter on an empty line at the "ai>" prompt, say the command, then stop talking: recording ends
after a short silence and the words go to the AI pilot as if typed. Speech is turned into text on this
laptop (Whisper, via faster-whisper), so it works on the drone's Wi-Fi too once the model is downloaded
(about 140 MB, the first time, needs internet).

Needs: pip install faster-whisper sounddevice
"""
import os
import queue
import re
import time

import numpy as np

RATE = 16000              # Whisper wants 16 kHz mono
BLOCK = 480               # 30 ms per block
MODEL = os.environ.get("VOICE_MODEL", "base.en")   # tiny.en is faster, small.en more accurate
END_SILENCE = 0.7         # s of quiet after speech that ends the recording
MAX_SECONDS = 10.0
WAIT_FOR_SPEECH = 6.0     # s to start talking before it gives up
# Drone words Whisper tends to get wrong
HINT = "Drone commands: take off, land, stop, kill, forward, back, left, right, up, down, hover, metres, seconds."

_model = None


def load():
    """Load the model now (a few seconds), so the first command isn't slow."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    return _model


def record():
    """Audio from the mic: starts when you talk, ends after END_SILENCE of quiet. None if nothing was said."""
    import sounddevice as sd
    q = queue.Queue()
    with sd.InputStream(samplerate=RATE, channels=1, blocksize=BLOCK, dtype="float32",
                        callback=lambda data, *_: q.put(data[:, 0].copy())):
        # The first 0.3 s is the room's noise level; speech is well above it
        noise = np.concatenate([q.get() for _ in range(10)])
        threshold = max(0.01, 3.5 * float(np.sqrt(np.mean(noise ** 2))))
        blocks, talking, quiet_since = [], False, None
        start = time.time()
        while True:
            b = q.get()
            loud = float(np.sqrt(np.mean(b ** 2))) > threshold
            now = time.time()
            if talking:
                blocks.append(b)
                if loud:
                    quiet_since = None
                else:
                    quiet_since = quiet_since or now
                    if now - quiet_since > END_SILENCE:
                        break
                if len(blocks) * BLOCK / RATE > MAX_SECONDS:
                    break
            elif loud:
                talking = True
                blocks = [noise[-BLOCK * 5:], b]   # keep the start of the first word
            elif now - start > WAIT_FOR_SPEECH:
                return None
    return np.concatenate(blocks)


def transcribe(audio):
    with np.errstate(all="ignore"):     # numpy on macOS warns about harmless overflow in Whisper's maths
        segments, _ = load().transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                        initial_prompt=HINT, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


def listen(say=print):
    """Record one command and return its text ('' if nothing was heard)."""
    load()
    say("  listening... (talk now, stop when done)")
    audio = record()
    if audio is None:
        say("  didn't hear anything")
        return ""
    t = time.time()
    text = transcribe(audio)
    say(f'  heard: "{text}"  ({time.time() - t:.1f} s)')
    return text


def direct_word(text, words):
    """'Stop.' / 'Land!' -> 'stop' / 'land' when the whole command is one of these words, else None."""
    w = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return w if w in words else None


if __name__ == "__main__":
    print(f"loading {MODEL}...", flush=True)
    t = time.time()
    load()
    print(f"ready in {time.time() - t:.1f} s. Enter to talk, Ctrl+C to quit.")
    while True:
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            break
        listen()
