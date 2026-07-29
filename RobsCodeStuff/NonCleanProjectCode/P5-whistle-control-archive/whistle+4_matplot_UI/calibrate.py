"""
Step 1 — Calibrate pitch bands to your whistle.

A live FFT spectrum is displayed so you can see exactly which frequency
your whistle hits. For each motor command you whistle for 2 seconds;
the script records the median peak frequency and auto-generates band
boundaries at the midpoints between adjacent pitches.

All interaction happens in the plot window itself (click it once to
give it focus, then press the indicated key) rather than the terminal.
This keeps the live spectrum genuinely live the whole time the script
runs, and avoids console-input quirks that differ between Windows and
macOS.

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
RECORD_SEC = 3
3                             # seconds to sample per command

audio_q    = queue.Queue(maxsize=4)  # bounded — drop stale frames, keep latest
latest     = {"chunk": np.zeros(CHUNK_SIZE, dtype="float32")}
ui_state   = {"prompt": ""}          # on-screen instruction text, set by callers
key_state  = {"key": None}           # most recent key press, consumed by _wait_for_key


def _audio_callback(indata, frames, t, status):
    if not audio_q.full():
        audio_q.put(indata.copy())


def _drain_to_latest(q, timeout=0.1):
    """Return the most recently queued chunk, discarding any backlog.

    Blocks briefly (up to `timeout`) for at least one chunk if the queue
    is currently empty; returns None if nothing arrives in that time.
    """
    try:
        chunk = q.get(timeout=timeout)
    except queue.Empty:
        return None
    while True:
        try:
            chunk = q.get_nowait()
        except queue.Empty:
            return chunk


# ── Live spectrum display ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.3))
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
prompt_txt = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                     fontsize=11, color="darkgreen", va="bottom",
                     wrap=True)
plt.tight_layout()


def _on_key(event):
    key_state["key"] = event.key


fig.canvas.mpl_connect("key_press_event", _on_key)


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
    prompt_txt.set_text(ui_state["prompt"])
    return line, peak_line, status_txt, prompt_txt


ani = animation.FuncAnimation(fig, _update_plot, interval=50, blit=True)


def _wait_for_key(valid_keys, prompt):
    """Show `prompt` on the plot and block (while keeping the mic/plot
    alive) until one of `valid_keys` is pressed *in the plot window*.

    The plot window must have focus (click it once) for key presses to
    register — this is a matplotlib requirement, not a bug.
    """
    ui_state["prompt"] = prompt
    key_state["key"] = None
    while True:
        chunk = _drain_to_latest(audio_q, timeout=0.02)
        if chunk is not None:
            latest["chunk"] = chunk.flatten()
        plt.pause(0.03)
        if key_state["key"] in valid_keys:
            pressed = key_state["key"]
            key_state["key"] = None
            return pressed


# Keys that count as "accept / go" — several aliases since key naming for
# the spacebar/enter can vary slightly across matplotlib GUI backends.
GO_KEYS = {"enter", " ", "space"}

# ── Calibration loop ──────────────────────────────────────────────────────────
detected_hz = {}


def _calibrate_command(command):
    """Block until the user whistles for RECORD_SEC and return median Hz.

    After sampling, shows the detected pitch and lets the user redo the
    recording (e.g. if background noise or a bad whistle threw it off)
    before accepting it.
    """
    while True:
        _wait_for_key(
            GO_KEYS,
            f"'{command}': click this window, then press ENTER or SPACE, "
            f"then whistle for {RECORD_SEC:.0f}s"
        )

        ui_state["prompt"] = f"Recording '{command}' — whistle now!"
        peaks = []
        deadline = time.time() + RECORD_SEC
        while time.time() < deadline:
            chunk = _drain_to_latest(audio_q)
            if chunk is not None:
                latest["chunk"] = chunk.flatten()
                hz, rms = detect(chunk)
                if rms >= AMPLITUDE_THRESHOLD:
                    peaks.append(hz)
            plt.pause(0.01)

        if not peaks:
            print("  No whistle detected — try again.")
            ui_state["prompt"] = (
                f"No whistle detected for '{command}' — "
                f"press ENTER/SPACE to retry"
            )
            continue

        median_hz = float(np.median(peaks))
        spread = float(np.std(peaks))
        print(f"  -> Detected {median_hz:.0f} Hz for '{command}'  "
              f"(n={len(peaks)} samples, std={spread:.0f} Hz)")

        key = _wait_for_key(
            GO_KEYS | {"r", "R"},
            f"'{command}': {median_hz:.0f} Hz detected — "
            f"ENTER/SPACE to accept, R to redo"
        )
        if key not in ("r", "R"):
            return median_hz
        print(f"  Redoing '{command}'...")


def _run_review_pass():
    """After all commands are calibrated, let the user redo any single
    one (e.g. if two pitches look suspiciously close together) before
    the bands are built and saved.
    """
    while True:
        print("\nDetected pitches:")
        keymap = {}
        for i, cmd in enumerate(COMMANDS, start=1):
            print(f"  [{i}] {cmd:<10} -> {detected_hz[cmd]:.0f} Hz")
            keymap[str(i)] = cmd

        listing = ", ".join(f"{k}={v}" for k, v in keymap.items())
        key = _wait_for_key(
            GO_KEYS | set(keymap.keys()),
            f"Review: ENTER/SPACE to finish, or press a number to redo "
            f"that command ({listing})"
        )
        if key in keymap:
            cmd = keymap[key]
            detected_hz[cmd] = _calibrate_command(cmd)
        else:
            break


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
    print("  All prompts appear on the plot window — click it once so")
    print("  it has keyboard focus, then follow the on-screen text.")
    print("=" * 55)

    for cmd in COMMANDS:
        detected_hz[cmd] = _calibrate_command(cmd)

    _run_review_pass()

    bands = _build_bands(detected_hz)

print("\nCalibrated pitch bands:")
for lo, hi, cmd in bands:
    print(f"  {lo:5d}-{hi:5d} Hz  ->  {cmd}")

with open(BANDS_FILE, "w") as f:
    json.dump(bands, f, indent=2)
print(f"\nSaved -> {BANDS_FILE}")
print("Run whistle.py to start driving.")

plt.close()