# pose-scratch

Train a pose classifier built entirely from scratch — no pretrained weights, no borrowed knowledge. Every parameter starts as random noise and must be learned entirely from the images you collect. The same gesture-to-motor pipeline as `pose-pytorch`, but a harder learning problem that reveals why transfer learning exists.

---

## How it differs from pose-pytorch

| | pose-pytorch | pose-scratch |
|---|---|---|
| **Starting weights** | ImageNet pretrained | Random (zeros / noise) |
| **Frozen layers** | Yes — backbone frozen | None — all layers train |
| **Images needed** | 30–50 per class | 100+ per class |
| **Epochs** | 20 | 40 |
| **Optimizer** | Adam | AdamW + cosine LR decay |
| **Expected accuracy** | 90–98% | 70–90% (dataset-dependent) |
| **Training time** | ~1 min | ~3–5 min |

The core lesson: a pretrained backbone already knows how to detect edges, textures, and shapes. A scratch network has to learn all of that too, from your ~500 images alone, which is a much harder ask.

---

## The architecture: ScratchCNN

`model.py` defines a simple 4-block convolutional network:

```
Input: 224 × 224 × 3

Conv(3→32)  + BN + ReLU + MaxPool  →  112 × 112 × 32
Conv(32→64) + BN + ReLU + MaxPool  →   56 ×  56 × 64
Conv(64→128)+ BN + ReLU + MaxPool  →   28 ×  28 × 128
Conv(128→128)+BN + ReLU + MaxPool  →   14 ×  14 × 128
AdaptiveAvgPool(1×1)               →    1 ×   1 × 128

Flatten → Linear(128→64) → ReLU → Dropout(0.4) → Linear(64→5)
```

**BatchNorm** after each conv stabilizes training with random init.
**Dropout** in the head reduces overfitting on small datasets.
**AdaptiveAvgPool** instead of a large Flatten keeps the parameter count low.

Total trainable parameters: ~340K (vs ~3.4M for MobileNetV2, most of which are frozen in pose-pytorch).

---

## Gesture classes

| Key | Class | Motor command |
|---|---|---|
| `1` | `backward` | Both motors reverse (−100, −100) |
| `2` | `forward` | Both motors forward (100, 100) |
| `3` | `stop` | Both motors off (0, 0) |
| `4` | `turn_left` | Left reverse, right forward (−60, 100) |
| `5` | `turn_right` | Left forward, right reverse (100, −60) |

---

## Files

| File | Purpose |
|---|---|
| `model.py` | ScratchCNN architecture definition |
| `collect_data.py` | Webcam tool — targets 100 images per class |
| `train.py` | Trains all layers from random init, saves best model |
| `pose_drive.py` | Real-time pose classification + motor control |
| `lelib.py` | SimpleLE wrapper around `legoeducation` |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
pip install -r requirements.txt
```

Set `SERIAL` in `pose_drive.py` to your Bluetooth card's serial number.

---

## Step 1: Collect training data

```bash
python collect_data.py
```

The overlay shows a counter for each class that turns green at 100 images. Aim to hit green on all five before training.

**Why so many?** With pretrained weights, 30 images suffice because the network already understands visual structure. From scratch, it has to learn everything — what a human silhouette looks like, what a raised arm looks like, how to distinguish your five gestures — all from your data alone.

**Tips:**
- Move between captures. Don't hold the same pose and spam Space — variety is more valuable than volume.
- Vary distance, lighting, and background if possible.
- Make the gestures exaggerated and distinct. Subtle differences between classes will confuse a scratch model far more than a pretrained one.

---

## Step 2: Train

```bash
python train.py
```

Training uses **AdamW** (Adam with weight decay) and a **cosine learning rate schedule** that starts at 5×10⁻⁴ and smoothly decays to near zero over 40 epochs. This helps the model settle into a good solution rather than oscillating.

**What healthy training looks like:**
```
Epoch  1/40  train=1.5821  val=1.4903  acc=28.0%  lr=5.00e-04
Epoch  5/40  train=1.1203  val=1.0841  acc=55.0%  lr=4.70e-04
Epoch 15/40  train=0.6201  val=0.7103  acc=75.0%  lr=3.45e-04
Epoch 30/40  train=0.3412  val=0.4821  acc=85.0%  lr=1.23e-04
Epoch 40/40  train=0.2103  val=0.4012  acc=88.0%  lr=5.00e-07
```
Both losses decrease together, validation accuracy climbs, and the gap between train and val stays reasonable.

**What overfitting looks like:**
```
Epoch 20/40  train=0.1500  val=0.9200  acc=65.0%
```
Train loss is very low but val loss is high — the model has memorized the training images rather than learning to generalize. Fix: collect more varied images and re-run.

When training finishes, `training_curve.png` is saved with loss and accuracy plots.

---

## Step 3: Drive

```bash
python pose_drive.py
```

The window shows the predicted gesture, confidence, motor speeds, and a probability bar for each class. Press **Q** to stop.

---

## Comparing the three pose approaches

| | pose-library | pose-pytorch | pose-scratch |
|---|---|---|---|
| **Model source** | MediaPipe (Google) | MobileNetV2 fine-tuned | Custom CNN from scratch |
| **Training required** | No | Yes (~30–50 img/class) | Yes (~100+ img/class) |
| **Control style** | Proportional (analog) | Discrete gestures | Discrete gestures |
| **What you learn** | API usage | Transfer learning | How CNNs actually work |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Accuracy stuck below 60% | Collect more images; check that gestures are clearly distinct |
| Val loss rising while train loss falls | Overfitting — more data and/or more augmentation |
| Model always predicts the same class | Class imbalance — balance your image counts |
| `pose_model.pt not found` | Run `train.py` first |
| Bluetooth connection error | Check `SERIAL` in `pose_drive.py` |
