# pose-library

Pose-controlled tank drive for a LEGO Education double motor, using your body as the joystick via [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

## What it does

`pose_drive.py` opens your webcam and uses **MediaPipe Pose** to detect 33 body landmarks in real time. It then maps the position of each wrist relative to its shoulder to a motor speed, giving you tank-drive control with your arms:

- **Left wrist above left shoulder** → left motor forward
- **Left wrist below left shoulder** → left motor backward
- **Right wrist above right shoulder** → right motor forward
- **Right wrist below right shoulder** → right motor backward
- **Wrists near shoulder height** → dead zone, motors stop

The further you raise or lower your wrist past your shoulder, the faster the motor runs (up to 100%). A small dead zone (±8% of frame height) prevents the motors from drifting when your arms are at rest.

The webcam image is **mirrored** so the control feels natural — your left arm controls the left motor.

## How the speed mapping works

MediaPipe returns landmark positions as fractions of the frame height (0 = top, 1 = bottom). Raising your wrist above your shoulder makes `wrist.y` smaller than `shoulder.y`, so:

```
offset = shoulder.y - wrist.y   # positive = wrist above shoulder
speed  = clamp((offset - dead_zone) × 250, -100, 100)
```

This gives a linear ramp from 0 to ±100% as your wrist moves roughly half a body-segment past your shoulder.

## Files

| File | Purpose |
|---|---|
| `main_pose_lib.py` | Main script — pose detection and drive loop |
| `requirements.txt` | Python dependencies |
| `pose_landmarker.task` | MediaPipe model file — auto-downloaded on first run |

`lelib.py` and `camlib.py` live in the project root and are shared across all examples.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Edit `pose_drive.py` and set `SERIAL` to your Bluetooth card's serial number:
   ```python
   SERIAL = 2279  # ← change this
   ```

3. Run:
   ```bash
   python main_pose_lib.py
   ```

## What happens at startup

**Model download** — on the first run the script fetches the MediaPipe pose landmarker model (~3 MB) and saves it as `pose_landmarker.task` in the same folder. Subsequent runs use the cached file. If your machine has SSL certificate issues (common with the python.org macOS installer) the download falls back to an unverified connection automatically.

**Camera selection** — the script scans for connected cameras. The terminal displays the following as it does so:

```
Scanning for cameras…
```

A window opens showing each of the cameras you can select with an assosiated numberkey. Press the numberkey of the camera you want to use.

A different window then opens showing the webcam feed with the pose skeleton drawn on top and the current motor speeds in the top-left corner. Press **Q** to stop.

## Controls

| Gesture | Effect |
|---|---|
| Raise left wrist above shoulder | Left motor forward |
| Lower left wrist below shoulder | Left motor backward |
| Raise right wrist above shoulder | Right motor forward |
| Lower right wrist below shoulder | Right motor backward |
| Arms relaxed at sides | Dead zone — motors stop |

## Dependencies

All pip-installable packages are listed in `requirements.txt`. The following standard-library modules are also used (no install needed): `ssl`, `urllib`, `subprocess`, `json`, `pathlib`.

## MediaPipe version note

This script uses the **MediaPipe Tasks API** (`mediapipe.tasks.vision.PoseLandmarker`), which is required for MediaPipe 0.10+. Older tutorials may show `mp.solutions.pose` — that API was removed in 0.10.x and will not work with current installs.