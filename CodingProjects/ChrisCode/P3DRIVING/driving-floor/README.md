# driving

A sketch-reading robot. A smartphone mounted on the LEGO double motor points straight down at the floor. You place hand-drawn command cards under the robot — forward arrow, backward arrow, left turn, right turn, stop — and the robot executes them in real time using a trained image classifier.

---

## How it works

```
Smartphone camera (pointing down)
        ↓  Wi-Fi stream
  collect_data.py  →  train.py  →  drive.py
    (capture cards)   (learn)      (read & drive)
```

1. **Collect** — Stream the phone camera to your laptop. Hold each sketch card under the camera and save labeled images.
2. **Train** — Fine-tune MobileNetV2 on your card images.
3. **Drive** — The robot reads the card under it, classifies the symbol, and drives accordingly. Swap cards to change command.

---

## Setup

### 1. Edit config.py

Open [config.py](config.py) — this is the only file you need to change:

```python
SERIAL = 1128                              # your LEGO Bluetooth card serial
CAMERA = "http://192.168.1.100:8080/video" # your phone's stream URL
CONFIDENCE_THRESHOLD = 0.70               # how certain before acting
```

### 2. Get the phone camera streaming

**Android — IP Webcam (recommended, free)**
1. Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) from the Play Store.
2. Open the app → scroll to the bottom → tap **Start server**.
3. Note the URL shown (e.g. `http://192.168.1.42:8080`).
4. Set `CAMERA = "http://192.168.1.42:8080/video"` in `config.py`.

**iOS — Camo**
1. Install [Camo](https://apps.apple.com/app/camo/id1451011458) on iPhone and Mac.
2. Connect by USB or Wi-Fi; Camo appears as a virtual camera (index `1` or `2`).
3. Set `CAMERA = 1` (or whichever index Camo uses) in `config.py`.

**Testing without a phone**
Set `CAMERA = 0` to use your laptop webcam and hold cards up to it instead.

### 3. Mount the phone on the robot

- Tape or rubber-band the phone to the top of the LEGO double motor assembly with the camera lens facing straight down.
- The camera should have a clear view of an A5/half-letter area directly beneath the robot.
- Make sure the phone and laptop are on the same Wi-Fi network.

### 4. Make the sketch cards

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

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 1: Collect training images

```bash
python collect_data.py
```

A window opens showing the phone camera feed live on your laptop.

**Workflow:**
1. Press a number key (`1`–`5`) to select a symbol class.
2. Place that sketch card flat on the floor under the robot/camera.
3. Press **Space** to capture. Aim for **40–60 images per class**.
4. Vary the card slightly between captures — shift it left/right, rotate a few degrees, move it closer or further.
5. Switch to the next class and repeat.
6. Press **Q** when done.

**Why vary?** The robot won't always stop perfectly centered over a card. Diversity in your training images (slight rotation, offset, distance) makes the model handle real driving conditions.

**Important — no horizontal flip augmentation:** A left arrow flipped is a right arrow. `train.py` deliberately omits horizontal flipping for this reason. Rotation augmentation (±20°) is used instead to handle card placement angle.

---

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

---

## Step 3: Drive

```bash
python drive.py
```

The laptop window shows the phone camera feed with:
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

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera` | Check IP in `config.py`; confirm phone and laptop are on same Wi-Fi; make sure the app's server is running |
| Robot ignores cards | Confidence below threshold — lower `CONFIDENCE_THRESHOLD` or collect more images |
| Left/right confused | The model may have learned the arrow pointing the wrong way — check your sketch cards are consistent |
| Works on desk but not while driving | Collect images with the robot moving slightly, or vary card position more during collection |
| Bluetooth connection error | Check `SERIAL` in `config.py`; make sure the LEGO hub is powered on |
