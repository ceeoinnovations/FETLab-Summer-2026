"""
Step 1 — Calibrate pitch bands to your whistle.

A live FFT spectrum is displayed so you can see exactly which frequency
your whistle hits. For each motor command you whistle for 2 seconds;
the script records the median peak frequency and auto-generates band
boundaries at the midpoints between adjacent pitches.

The result is saved to pitch_bands.json. whistle.py loads this file
automatically on startup.

macOS note: you may need to grant Terminal microphone access in
  System Settings → Privacy & Security → Microphone.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import queue
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import (SAMPLE_RATE, CHUNK_SIZE, MOTOR_MAP,
                    WHISTLE_MIN_HZ, WHISTLE_MAX_HZ, AMPLITUDE_THRESHOLD)
from pitch import detect, spectrum_snapshot
from miclib import pick_mic

BANDS_FILE = Path(__file__).parent / "pitch_bands.json"
COMMANDS   = list(MOTOR_MAP.keys())          # calibrate all five commands
RECORD_SEC = 2.0                             # seconds to sample per command

audio_q    = queue.Queue()
latest     = {"chunk": np.zeros(CHUNK_SIZE, dtype="float32")}


def _audio_callback(indata, frames, t, status):
    audio_q.put(indata.copy())


# ── Live spectrum display ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3))
from pitch import _VIS_FREQS
line,      = ax.plot(_VIS_FREQS, np.zeros(len(_VIS_FREQS)), lw=1.5, color="steelblue")
peak_line  = ax.axvline(x=0, color="crimson", lw=1.5, label="peak")
ax.set_xlim(WHISTLE_MIN_HZ, WHISTLE_MAX_HZ)
ax.set_ylim(0, 0.3)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_title("Live spectrum — whistle to see your pitch")
status_txt = ax.text(0.02, 0.92, "", transform=ax.transAxes,
                     fontsize=10, color="navy", va="top")
plt.tight_layout()


def _update_plot(_):
    chunk = latest["chunk"]
    freqs, amps = spectrum_snapshot(chunk)
    line.set_ydata(amps)
    hz, rms = detect(chunk)
    if rms >= AMPLITUDE_THRESHOLD:
        peak_line.set_xdata([hz])
        status_txt.set_text(f"{hz:.0f} Hz   rms={rms:.3f}")
    else:
        status_txt.set_text("silence")
    return line, peak_line, status_txt


ani = animation.FuncAnimation(fig, _update_plot, interval=50, blit=True)

# ── Calibration loop ──────────────────────────────────────────────────────────
detected_hz = {}

def _calibrate_command(command):
    """Block until the user whistles for RECORD_SEC and return median Hz."""
    input(f"\n  [Enter] then whistle for  '{command}'  ({RECORD_SEC:.0f} s)... ")
    peaks = []
    deadline = time.time() + RECORD_SEC
    while time.time() < deadline:
        try:
            chunk = audio_q.get(timeout=0.1)
            latest["chunk"] = chunk.flatten()
            hz, rms = detect(chunk)
            if rms >= AMPLITUDE_THRESHOLD:
                peaks.append(hz)
            plt.pause(0.01)
        except queue.Empty:
            pass
    if not peaks:
        print(f"  No whistle detected — try again.")
        return _calibrate_command(command)
    median_hz = float(np.median(peaks))
    print(f"  → Detected {median_hz:.0f} Hz for '{command}'")
    return median_hz


def _build_bands(detected: dict) -> list:
    """
    Sort commands by ascending frequency, then split the whistle range
    at midpoints between adjacent detected frequencies.
    """
    ordered = sorted(detected.items(), key=lambda kv: kv[1])
    bands   = []
    lo      = WHISTLE_MIN_HZ
    for i, (cmd, hz) in enumerate(ordered):
        hi = ((hz + ordered[i + 1][1]) / 2) if i < len(ordered) - 1 else WHISTLE_MAX_HZ
        bands.append((round(lo), round(hi), cmd))
        lo = hi
    return bands


mic_device = pick_mic()

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                    blocksize=CHUNK_SIZE, dtype="float32",
                    device=mic_device,
                    callback=_audio_callback):

    plt.show(block=False)
    print("=" * 55)
    print("  Whistle Calibration")
    print("  Watch the spectrum window to see your whistle frequency.")
    print("=" * 55)

    for cmd in COMMANDS:
        detected_hz[cmd] = _calibrate_command(cmd)

    bands = _build_bands(detected_hz)

print("\nCalibrated pitch bands:")
for lo, hi, cmd in bands:
    print(f"  {lo:5d}–{hi:5d} Hz  →  {cmd}")

with open(BANDS_FILE, "w") as f:
    json.dump(bands, f, indent=2)
print(f"\nSaved → {BANDS_FILE}")
print("Run whistle.py to start driving.")

plt.close()
