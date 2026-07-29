# driving

A sketch-reading robot. A smartphone mounted on the LEGO double motor points straight down at the floor. You place hand-drawn command cards under the robot — forward arrow, backward arrow, left turn, right turn, stop — and the robot executes them in real time using a trained image classifier.

## How it works

```
Smartphone camera
        ↓ 
(Via Wi-Fi stream)
        ↓  
  collect_data.py  →  train.py  →  drive.py
    (collect)        (learn)    (read & drive)
```

1. **Collect** — Stream the phone camera to your laptop. Hold each sketch card under the camera and save labeled images.
2. **Train** — Fine-tune MobileNetV2 on your card images.
3. **Read & Drive** — The robot reads the card under it, classifies the symbol, and drives accordingly. Swap cards to change command.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Camo Studio

1. Install [Camo](https://apps.apple.com/app/camo/id1451011458) on your phone and computer.
2. In the laptop application, click the dropdown menu under "Devices" and select "+ Pair Camo Camera". You should see a QR code.
3. In the mobile application, click the Wi-Fi symbol in the top right. Scan the QR code on the computer to stream video feed from your phone to your camera.


### 3. Edit config.py

Open [config.py](config.py) — this is the only file you need to change.

```python
SERIAL = 1227                              # your LEGO Bluetooth card serial
CONFIDENCE_THRESHOLD = 0.70                # how certain before acting
```

### 4. Mount the phone on the robot

- Use Legos to mount the phone.
- The camera should have a clear view of the ground in front of it.
- Make sure the phone and laptop are on the same Wi-Fi network.

### 5. Make the sketch cards

Draw each symbol on a separate piece of white paper or card stock, large enough to fill the camera's view:

| Symbol | Meaning | Motor command |
|---|---|---|
| ↑ (arrow up) | Forward | Both wheels forward (100, 100) |
| ↓ (arrow down) | Backward | Both wheels reverse (−100, −100) |
| ← (arrow left) | Turn left | Left reverse, right forward (−70, 100) |
| → (arrow right) | Turn right | Left forward, right reverse (100, −70) |
| ✕ or STOP | Stop | Both wheels stop (0, 0) |

**Tips for reliable recognition:**
- Draw symbols large, bold, and centered — at least half the card area.
- Use a thick black marker on white paper.
- Keep the style consistent across cards — the model learns your handwriting.
- Laminate or use card stock so cards stay flat on the floor.


## Step 1: Collect training images

Have your phone already streaming video to your computer via Camo Studio.

```bash
python collect_data.py
```

The Camo Studio camera feed should automatically be connected to. An overlay should pop-up that allows you to start data collection.

> **Note on the overlay:** the symbol name / counts / instructions text is drawn only on the preview window, on a copy of the frame. The image actually written to disk is the clean, un-overlaid frame — so the UI text never ends up baked into your training data.

**Workflow:**
1. Press a number key (`1`–`5`) to select a symbol class.
2. Place that sketch card flat on the floor under the robot/camera.
3. Capture frames using either method below. Aim for **40–60 images per class**.
4. Vary the card slightly between captures — shift it left/right, rotate a few degrees, move it closer or further.
5. Switch to the next class and repeat.
6. Press **Q** when done.

**Capture controls:**

| Key | Behavior |
|---|---|
| **Space** | Starts a timed burst: first waits `PRE_RECORD_DELAY_SEC` seconds so you can get the card positioned, shown as a "GET READY …s" countdown, then auto-captures frames at `TIMER_CAPTURE_HZ` for `TIMER_DURATION_SEC` seconds (shown as "RECORDING …s left"), then stops on its own. |
| **Enter** | Tap once to capture a single frame immediately. Hold it down to keep capturing frames for as long as it's held. |

All adjustable at the top of `collect_data.py`:

```python
TIMER_CAPTURE_HZ       = 10    # frames captured per second during a Space burst
TIMER_DURATION_SEC     = 3.0   # how long a Space burst lasts, in seconds
PRE_RECORD_DELAY_SEC   = 3.0   # seconds to position the card before capture starts
ENTER_MIN_INTERVAL_SEC = 0.15  # time between captured frames when holding down enter
```

You can't switch classes or start a new Space burst while a countdown or timed burst is already running — let it finish before doing either.

**Why vary?** The robot won't always stop perfectly centered over a card. Diversity in your training images (slight rotation, offset, distance) makes the model handle real driving conditions.

**Important — no horizontal flip augmentation:** A left arrow flipped is a right arrow. `train.py` deliberately omits horizontal flipping for this reason. Rotation augmentation (±20°) is used instead to handle card placement angle.

## Step 2: Train

```bash
python train.py
```

Expected output:
```
Classes found: ['backward', 'forward', 'left', 'right', 'stop']
Total images : 250
Training on : mps

Epoch  1/20  train=1.4231  val=1.1823  acc=55.0%
           ↑ best model saved → symbol_model.pt
...
Epoch 20/20  train=0.0821  val=0.1203  acc=96.0%
```

Validation accuracy above 90% means the model is ready. `training_curve.png` is saved for review.

## Step 3: Drive

Have your phone already streaming video to your computer via Camo Studio.

```bash
python drive.py
```

A laptop window will open showing the phone camera feed with:
- **Top**: predicted symbol and confidence percentage (green = acting, orange = uncertain/stopped)
- **Second line**: left and right motor speeds being sent
- **Bottom**: a probability bar for each class

**How commands work:**
- Place a card under the robot → it reads the symbol and drives continuously.
- Swap to a different card → the command changes immediately.
- Remove the card or show a blank surface → confidence drops below threshold → motors stop.
- The robot drives until told otherwise — it's reading continuously at ~30 Hz.

**Adjusting sensitivity:** If the robot acts on uncertain readings, raise `CONFIDENCE_THRESHOLD` in `config.py` (e.g. 0.80). If it's too hesitant, lower it (e.g. 0.60).

Press **Q** to quit and stop the motors.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera` | Check IP in `config.py`; confirm phone and laptop are on same Wi-Fi; make sure the app's server is running |
| Robot ignores cards | Confidence below threshold — lower `CONFIDENCE_THRESHOLD` or collect more images |
| Left/right confused | The model may have learned the arrow pointing the wrong way — check your sketch cards are consistent |
| Works on desk but not while driving | Collect images with the robot moving slightly, or vary card position more during collection |
| Bluetooth connection error | Check `SERIAL` in `config.py`; make sure the LEGO hub is powered on |
