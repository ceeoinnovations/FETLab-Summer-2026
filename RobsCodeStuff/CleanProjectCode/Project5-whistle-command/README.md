# whistle-commands

Drive the LEGO double motor by whistling at different pitches. No training data, no neural network — just the FFT algorithm applied to your microphone in real time. Each whistle pitch maps to a different motor command: forward, backward, left, right, or stop.

---

## How it works

```
calibrate.py  →  whistle.py
  (tune bands)   (drive)
```

The computer's microphone captures audio in 46 ms windows. Each window is analyzed with the **Fast Fourier Transform (FFT)** to find the dominant frequency. That frequency is looked up in a table of pitch bands to determine the motor command.

---

## The algorithm: FFT pitch detection

Sound is a pressure wave — a 1-D signal of amplitude over time. The FFT decomposes that signal into its frequency components, telling you how much of each frequency is present.

```
Time domain:  ████▄▄▄▄████▄▄▄▄████  (raw microphone samples)
      ↓  FFT
Freq domain:  ...| peak at 1800 Hz |...  (spectrum)
```

For a whistle, almost all the energy concentrates at a single frequency (the pitch), so the FFT peak unambiguously identifies the note you're whistling.

**Implementation details (in `pitch.py`):**

1. **Hanning window** — multiply the audio chunk by a bell-shaped curve before the FFT. This prevents "spectral leakage" — energy smearing from sharp chunk edges into neighboring frequency bins.

2. **rfft** — since audio is real-valued, `numpy.fft.rfft` gives the one-sided spectrum (0 Hz to Nyquist), halving the computation.

3. **Frequency resolution** — with `CHUNK_SIZE=2048` samples at 44100 Hz, each FFT bin is `44100/2048 ≈ 21.5 Hz` wide. That's more than precise enough to distinguish whistle pitches.

4. **RMS threshold** — Root Mean Square amplitude measures overall loudness. If RMS < `AMPLITUDE_THRESHOLD`, the frame is silence and motors stop, regardless of the FFT result.

5. **Only look in 300–5000 Hz** — this band covers all realistic whistles and ignores most background noise (hum, HVAC, speech).

---

## Files

| File | Purpose |
|---|---|
| `pitch.py` | FFT pitch detection — `detect()`, `freq_to_command()`, `spectrum_snapshot()` |
| `config.py` | Serial number, audio parameters, default pitch bands, motor map |
| `calibrate.py` | Live spectrum display + interactive calibration → `pitch_bands.json` |
| `whistle.py` | Real-time pitch → motor control |
| `lelib.py` | SimpleLE wrapper |
| `requirements.txt` | Dependencies (`sounddevice`, `numpy`, `matplotlib`, `legoeducation`) |

---

## Setup

All dependencies were installed with the project venv. The only requirement is microphone access:

> **macOS:** Go to **System Settings → Privacy & Security → Microphone** and enable access for **Terminal** (or your IDE).

Set `SERIAL` in `config.py` to your Bluetooth card serial number.

---

## Step 1: Calibrate

```bash
python calibrate.py
```

As soon as you start the code, you should see the terminal list out the available audio sorces that your system has access to. Select the one you would like to use.

```
Available microphone inputs:
  [1] MacBook Pro Microphone ← default
  [3] Camo Microphone
  [4] Rob Phone Microphone
  [5] Microsoft Teams Audio
```

A live spectrum window opens showing the FFT of your microphone input in real time. The red vertical line marks the detected peak frequency.

For each motor command, the window prompts you to whistle for 3 seconds. After each recording you can press **R** to redo the recording or you can press **Enter** or **Space** to move to the next motor command. 

After all five commands are calibrated, the script auto-generates pitch bands by placing boundaries at the midpoints between adjacent detected frequencies:

```
Calibrated pitch bands:
    300–  821 Hz  →  backward
    821– 1312 Hz  →  left
   1312– 2145 Hz  →  forward
   2145– 3672 Hz  →  right
   3672– 5000 Hz  →  stop
```

You can press **Enter** or **Space** to confirm those pitch bands or you can press the number of the motor command to redo that pitch band.

These are saved to `pitch_bands.json`.

**Tips for a good calibration:**
- Whistle each command at a clearly distinct pitch — spread them out across your range.
- Sustain the pitch steadily for the full 2 seconds (don't slide).
- Calibrate in the same room where you'll drive — background noise levels matter.
- If you can't whistle, a physical slide whistle or recorder works perfectly.

---

## Step 2: Drive

```bash
python whistle.py
```

Output while driving:
```
   Hz     rms  command       L      R
────────────────────────────────────────────
  1847   0.043  forward    +100   +100
  1821   0.051  forward    +100   +100
     —   0.004  stop         +0     +0   ← silence
   762   0.038  backward   -100   -100
```

- **Hz** — detected dominant frequency (— when silent)
- **rms** — microphone loudness (0–1); below `AMPLITUDE_THRESHOLD` → silence
- **command** — the mapped motor command
- **L / R** — motor speeds being sent

Motor commands only update when the command **changes** — this avoids flooding the Bluetooth link with identical packets.

Press **Ctrl+C** to stop.

---

## Default pitch bands

If you skip calibration, these defaults from `config.py` are used:

| Band | Command |
|---|---|
| 300–900 Hz | backward |
| 900–1500 Hz | left |
| 1500–2500 Hz | forward |
| 2500–5000 Hz | right |
| silence | stop |

---

## Tuning tips

| Problem | Fix |
|---|---|
| Motors trigger on background noise | Raise `AMPLITUDE_THRESHOLD` in `config.py` |
| Robot doesn't respond to whistling | Lower `AMPLITUDE_THRESHOLD` |
| Commands bleed into each other | Re-run `calibrate.py` with more distinct pitches |
| "No whistle detected" during calibration | Whistle louder or closer to the mic |
| Wants more commands | Add entries to `MOTOR_MAP` in `config.py` and re-calibrate |
