"""
Step 2 — Train the behavior-cloning steering model.

Loads (image, left_speed, right_speed) pairs from data/labels.csv and
data/images/, normalizes motor speeds to -1..1, and fine-tunes the
regression head from model.py (frozen MobileNetV2 backbone).

Saves the best model to drive_model.pt and a training curve plot.
"""

import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from model import build_model
from config import MOTOR_SPEED_MIN, MOTOR_SPEED_MAX, IMG_SIZE

DATA_DIR = Path(__file__).parent / "data"
IMG_DIR = DATA_DIR / "images"
LABELS_CSV = DATA_DIR / "labels.csv"
MODEL_OUT = "drive_model.pt"
EPOCHS = 20
BATCH = 30
LR = 1e-3


def normalize_speed(s):
    # map MOTOR_SPEED_MIN..MAX -> -1..1
    mid = (MOTOR_SPEED_MAX + MOTOR_SPEED_MIN) / 2
    span = (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN) / 2
    return (s - mid) / span


class DriveDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fname, left, right = self.rows[idx]
        img = Image.open(IMG_DIR / fname).convert("RGB")
        img = self.transform(img)
        target = torch.tensor(
            [normalize_speed(float(left)), normalize_speed(float(right))],
            dtype=torch.float32,
        )
        return img, target


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # No horizontal flip — flipping would invert left/right speed labels.
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

with open(LABELS_CSV) as f:
    reader = csv.reader(f)
    next(reader)  # header
    rows = [row for row in reader]

print(f"Total samples: {len(rows)}")

n_val = max(1, int(0.2 * len(rows)))
n_train = len(rows) - n_val
full_ds = DriveDataset(rows, transform)
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                  generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

device = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}\n")

model = build_model().to(device)
for p in model.features.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            val_loss += criterion(model(imgs), targets).item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    print(f"Epoch {epoch + 1:2d}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({"state_dict": model.state_dict()}, MODEL_OUT)
        print(f"           ↑ best model saved → {MODEL_OUT}")

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.title("Steering Regression Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
