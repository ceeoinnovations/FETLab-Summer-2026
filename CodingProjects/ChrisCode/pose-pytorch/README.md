# pose-pytorch

Train your own pose classifier with PyTorch and use it to drive a LEGO Education double motor with your body. Unlike `pose-library` (which uses MediaPipe's pre-built model), this project teaches you to build and train the model yourself using **transfer learning** on images you capture.

---

## How it works

The pipeline has three stages:

```
collect_data.py  →  train.py  →  main_pose_train.py
  (capture)         (learn)           (drive)
```

1. **Collect** — You stand in front of your webcam and press keys to save labeled images of each gesture into folders.
2. **Train** — A pretrained MobileNetV2 CNN is fine-tuned on your images. Only its final layer changes; the rest already knows how to see.
3. **Drive** — The trained model classifies your pose in real time and maps each gesture to a motor command.

---

## The model: MobileNetV2 + transfer learning

`model.py` wraps `torchvision.models.mobilenet_v2`, which was pretrained on ImageNet (1.2 million images, 1000 classes). Its convolutional layers already detect edges, textures, shapes, and body parts — so you don't need thousands of images.

We replace only the final fully-connected layer with a new one that outputs probabilities for our 5 gesture classes:

```
[pretrained MobileNetV2 backbone]  →  [new Linear(1280, 5)]  →  softmax
```

During training, the backbone weights are **frozen** — only the new head is updated. This means:
- Training takes seconds per epoch, not hours.
- 30–50 images per class is enough.
- You don't need a GPU (Apple MPS or CPU both work fine).

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
| `model.py` | MobileNetV2 architecture + gesture class names |
| `collect_data.py` | Webcam tool to capture and label training images |
| `train.py` | Trains the model, saves weights, plots loss curves |
| `main_pose_train.py` | Loads trained model, classifies pose, drives motors |
| `data/` | Training images, one sub-folder per gesture class |
| `pose_model.pt` | Saved model weights — created by `train.py` |
| `training_curve.png` | Loss/accuracy plot — created by `train.py` |
| `requirements.txt` | Python dependencies |

`lelib.py` and `camlib.py` live in the project root and are shared across all examples.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> PyTorch install note: if the above fails for `torch`, install it from [pytorch.org](https://pytorch.org/get-started/locally/) choosing your OS and "pip". On Apple Silicon Macs, MPS acceleration is available automatically.

### 2. Set your serial number

Open `main_pose_train.py` and change the `SERIAL` constant to your Bluetooth card's number:

```python
SERIAL = 2279  # ← change this
```

---

## Step 1: Collect training data

```bash
python collect_data.py
```

A camera picker window opens first — press the number key for the camera you want to use. Then the main capture window appears showing your webcam feed. The overlay shows which class is selected and how many images you've saved so far.

**Workflow:**
1. Press a number key (`1`–`5`) to select a gesture class.
2. Strike that pose in front of the camera.
3. Press **Space** to capture a frame. Repeat until you have at least **30–50 images** for that class.
4. Switch to the next class and repeat.
5. Press **Q** when done.

**Tips for good data:**
- Vary your distance from the camera, lighting, and exact body position. Diversity makes the model more robust.
- Move a little between captures — don't just spam Space on a frozen pose.
- Make the gestures distinct. For `stop`, keep arms neutral at your sides. For `forward`, raise both arms. The more different each class looks, the easier it is to classify.
- Try to collect a similar number of images per class (imbalance hurts accuracy).

Images are saved to `pose-pytorch/data/<class_name>/00000.jpg`, `00001.jpg`, etc. — always inside the `pose-pytorch` folder regardless of which directory you run the script from. You can delete bad images manually.

---

## Step 2: Train the model

```bash
python train.py
```

The script will print progress for each epoch:

```
Classes found: ['backward', 'forward', 'stop', 'turn_left', 'turn_right']
Total images : 200
Training on : mps

Epoch  1/20  train=1.4821  val=1.2043  acc=62.5%
           ↑ best model saved → pose_model.pt
Epoch  2/20  train=0.9134  val=0.7201  acc=75.0%
           ↑ best model saved → pose_model.pt
...
Epoch 20/20  train=0.1823  val=0.2104  acc=95.0%
```

**What to watch for:**
- Both `train` and `val` loss should decrease together. If `val` stops dropping while `train` keeps falling, the model is overfitting — collect more images.
- Accuracy above ~90% on validation means the model is ready to drive.
- The best checkpoint (lowest validation loss) is saved automatically.

When training finishes, `training_curve.png` is saved showing loss and accuracy over epochs. The saved model file `pose_model.pt` contains both the weights and the class names.

---

## Step 3: Drive with your pose

```bash
python main_pose_train.py
```

The webcam window shows:
- **Top left**: the predicted gesture class and confidence percentage.
- **Second line**: the resulting left and right motor speeds.
- **Bottom**: a probability bar for each class, so you can see how confident the model is in real time.

Strike a pose to drive. Press **Q** or close the window to stop the motors and quit.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Low accuracy after training | Collect more images; aim for 50+ per class with varied poses |
| Model always predicts the same class | You likely have a class imbalance — balance the image counts |
| Jerky motor response | The model is uncertain; try holding poses more steadily, or collect more data |
| `pose_model.pt not found` | Run `train.py` before `main_pose_train.py` |
| Bluetooth connection error | Check `SERIAL` matches your card; make sure the device is on |
