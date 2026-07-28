"""
Step 2 — Train the behavior-cloning steering model.

Loads (image, left_speed, right_speed) pairs from data/labels.csv and
data/images/, normalizes motor speeds to -1..1, and fine-tunes the
regression head (plus the last couple of backbone blocks) from
model.py.

Saves the best model to drive_model.pt and a training curve plot.

Changes from the original version, and why:

1. Train/val split is now done PER DRIVE (session_id), not per frame.
   At 10Hz, neighboring frames are nearly identical, so a random
   per-frame split leaks near-duplicates across train/val and makes
   validation loss look better than true generalization to a new
   drive. Splitting whole sessions instead gives an honest estimate.

2. Training batches are now sampled with a turning-aware weight, so
   the model sees turning examples far more often than the raw data
   would otherwise allow. Most collected frames are "drive straight"
   or "stopped" (often 60-90% of the dataset) — the original random
   sampling essentially taught the model to always play it safe and
   predict "small/no turn". This does NOT drop or duplicate images on
   disk, it only changes how often each one is drawn during training.

3. The last 2 backbone blocks are unfrozen (at a lower learning rate
   than the head) instead of freezing the whole backbone. A fully
   frozen ImageNet backbone was never trained to localize "where
   (left/right) is the interesting thing in this frame", which is
   exactly the cue steering direction depends on.
"""

import csv
import random
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4          # smaller LR for the unfrozen backbone layers
UNFREEZE_LAST_N_BLOCKS = 2  # how many of the final MobileNetV2 blocks to fine-tune
VAL_FRACTION = 0.2
SEED = 42


def normalize_speed(s):
    # map MOTOR_SPEED_MIN..MAX -> -1..1
    mid = (MOTOR_SPEED_MAX + MOTOR_SPEED_MIN) / 2
    span = (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN) / 2
    return (s - mid) / span


class DriveDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows  # list of (fname, left, right)
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

# ── Load labels, grouped by drive/session ────────────────────────────────────
with open(LABELS_CSV) as f:
    reader = csv.reader(f)
    header = next(reader)
    has_session = "session_id" in header
    all_rows = [row for row in reader]

print(f"Total samples: {len(all_rows)}")

rows_by_session = defaultdict(list)
if has_session:
    for row in all_rows:
        fname, left, right, session_id = row[0], row[1], row[2], row[3]
        rows_by_session[session_id].append((fname, left, right))
else:
    # Older dataset collected before collect_data.py wrote session_id.
    # Falling back to one big session means the train/val split below is
    # just a random per-frame split again (with the leakage that implies).
    # Prefer re-collecting with the updated collect_data.py if you can.
    print("WARNING: labels.csv has no session_id column — train/val split "
          "will not be session-aware for this dataset. Re-collect with the "
          "updated collect_data.py to fix this.")
    rows_by_session["0"] = [(r[0], r[1], r[2]) for r in all_rows]

session_ids = list(rows_by_session.keys())
rng = random.Random(SEED)
rng.shuffle(session_ids)

if len(session_ids) < 2:
    # Can't hold out a whole drive if there's only one (or zero) — fall back
    # to a random per-frame split so the script still runs, but this means
    # the usual per-frame leakage caveat applies. Collect at least a
    # couple of separately-marked drives to get an honest validation split.
    print("WARNING: only one session found — falling back to a random "
          "per-frame train/val split (val loss will be optimistic).")
    only_rows = next(iter(rows_by_session.values()))
    idx = list(range(len(only_rows)))
    rng.shuffle(idx)
    n_val = max(1, int(VAL_FRACTION * len(idx)))
    val_idx, train_idx = set(idx[:n_val]), set(idx[n_val:])
    train_rows = [only_rows[i] for i in train_idx]
    val_rows = [only_rows[i] for i in val_idx]
    train_sessions, val_sessions = {"n/a"}, {"n/a"}
else:
    n_val_sessions = max(1, int(VAL_FRACTION * len(session_ids)))
    n_val_sessions = min(n_val_sessions, len(session_ids) - 1)  # always leave >=1 for training
    val_sessions = set(session_ids[:n_val_sessions])
    train_sessions = set(session_ids[n_val_sessions:])

    train_rows, val_rows = [], []
    for sid, rows in rows_by_session.items():
        (val_rows if sid in val_sessions else train_rows).extend(rows)

print(f"Sessions: {len(session_ids)} total -> {len(train_sessions)} train / "
      f"{len(val_sessions)} val")
print(f"Frames:   {len(train_rows)} train / {len(val_rows)} val")

train_ds = DriveDataset(train_rows, transform)
val_ds = DriveDataset(val_rows, transform)

# ── Turning-aware sampling for the TRAINING set only ─────────────────────────
# Validation stays as naturally distributed as possible so its loss still
# means "how well does this match a real drive", not "how well does it match
# an artificially rebalanced one".
turn_mag = [abs(float(l) - float(r)) for _, l, r in train_rows]
# weight grows with how sharp the turn is, with a floor of 1 so straight/stopped
# frames are still seen (just less disproportionately than in the raw data).
sample_weights = [1.0 + 3.0 * m for m in turn_mag]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

device = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}\n")

model = build_model().to(device)

# Freeze the whole backbone, then re-unfreeze just the last few blocks so
# they can adapt to this task (in particular, learning to preserve *where*
# in the frame the target is, not just *whether* it's present).
for p in model.features.parameters():
    p.requires_grad = False
backbone_finetune_params = []
for block in list(model.features.children())[-UNFREEZE_LAST_N_BLOCKS:]:
    for p in block.parameters():
        p.requires_grad = True
        backbone_finetune_params.append(p)

head_params = list(model.classifier.parameters())

optimizer = torch.optim.Adam([
    {"params": head_params, "lr": LR_HEAD},
    {"params": backbone_finetune_params, "lr": LR_BACKBONE},
])
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
plt.title("Steering Regression Loss (MSE) — session-split, turn-weighted")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
