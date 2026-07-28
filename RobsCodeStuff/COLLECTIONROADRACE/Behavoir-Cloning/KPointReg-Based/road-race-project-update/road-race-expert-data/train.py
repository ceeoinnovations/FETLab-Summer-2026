"""
Train the end-to-end behavior-cloning steering model.

This project no longer collects its own data (see the removed
collect_data.py) — it trains purely on data EXPORTED from elsewhere, most
directly from option1_keypoint_regression's drive.py running with
EXPORT_EXPERT_DATA = True in its config.py. That export produces the same
(image, left_speed, right_speed, session_id) format collect_data.py used
to produce by hand, just generated automatically by option1's own
perception+control loop driving the car instead of a human joystick.

Why train on that instead of raw human joystick data: option1's control
loop is a deterministic, continuous proportional controller — for a given
scene it always outputs the same, smoothly-varying response, with none of
the noise, inconsistency, or timing lag a human's hand introduces. That
makes it a cleaner "teacher" to imitate than the human demonstrations the
very first version of this project trained on.

Usage:
    python train.py                                   (uses data/images, data/labels.csv)
    python train.py <images_dir> <labels_csv>          (explicit paths)

Loads (image, left_speed, right_speed) pairs, normalizes motor speeds to
-1..1, and fine-tunes model.py's regression head (plus the last couple of
backbone blocks) against them.

Saves the best model to drive_model.pt and a training curve plot.
"""

import sys
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

MODEL_OUT = "drive_model.pt"
EPOCHS = 30
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
    def __init__(self, img_dir, rows, transform):
        self.img_dir = Path(img_dir)
        self.rows = rows  # list of (fname, left, right)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fname, left, right = self.rows[idx]
        img = Image.open(self.img_dir / fname).convert("RGB")
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


def load_rows_with_sessions(labels_csv):
    """Reads labels.csv (filename, left_speed, right_speed, session_id).
    Older exports without a session_id column are treated as one big
    session — still works, just with a random-split fallback below."""
    with open(labels_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    has_session = "session_id" in header
    session_idx = header.index("session_id") if has_session else None

    rows_by_session = defaultdict(list)
    for row in rows:
        fname, left, right = row[0], row[1], row[2]
        sid = row[session_idx] if has_session else "0"
        rows_by_session[sid].append((fname, left, right))
    return rows_by_session, has_session


def session_aware_split(rows_by_session, has_session):
    """Splits by whole session/drive rather than by individual frame, so
    validation isn't just near-duplicate neighboring frames of a training
    frame — same reasoning used throughout this whole project's other
    training scripts."""
    session_ids = list(rows_by_session.keys())
    rng = random.Random(SEED)
    rng.shuffle(session_ids)

    if not has_session:
        print("NOTE: labels.csv has no session_id column — falling back to "
              "a random per-frame split (val loss will be a bit optimistic). "
              "Data exported from option1's drive.py includes session_id; "
              "this fallback is for older/hand-edited datasets only.")

    if len(session_ids) < 2:
        only_rows = next(iter(rows_by_session.values()))
        idx = list(range(len(only_rows)))
        rng.shuffle(idx)
        n_val = max(1, int(VAL_FRACTION * len(idx)))
        val_idx, train_idx = set(idx[:n_val]), set(idx[n_val:])
        return [only_rows[i] for i in train_idx], [only_rows[i] for i in val_idx]

    n_val_sessions = max(1, int(VAL_FRACTION * len(session_ids)))
    n_val_sessions = min(n_val_sessions, len(session_ids) - 1)  # always leave >=1 for training
    val_sessions = set(session_ids[:n_val_sessions])
    train_sessions = set(session_ids[n_val_sessions:])

    train_rows, val_rows = [], []
    for sid, rows in rows_by_session.items():
        (val_rows if sid in val_sessions else train_rows).extend(rows)
    return train_rows, val_rows


def main(img_dir, labels_csv):
    rows_by_session, has_session = load_rows_with_sessions(labels_csv)
    total = sum(len(r) for r in rows_by_session.values())
    print(f"Total samples: {total}")

    train_rows, val_rows = session_aware_split(rows_by_session, has_session)
    print(f"Sessions: {len(rows_by_session)} total")
    print(f"Frames:   {len(train_rows)} train / {len(val_rows)} val")

    train_ds = DriveDataset(img_dir, train_rows, transform)
    val_ds = DriveDataset(img_dir, val_rows, transform)

    # Turning-aware sampling for the TRAINING set only — most driving is
    # "go straight/stop", so without this the model would rarely see a
    # strong turn example relative to how often it needs to actually
    # perform one. Validation stays naturally distributed.
    turn_mag = [abs(float(l) - float(r)) for _, l, r in train_rows]
    sample_weights = [1.0 + 3.0 * m for m in turn_mag]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    model = build_model().to(device)

    # Freeze the whole backbone, then re-unfreeze just the last few blocks
    # so they can adapt to this specific task (in particular, learning to
    # preserve *where* in the frame something interesting is, not just
    # *whether* it's present — a fully-frozen ImageNet backbone was never
    # trained to keep that information through to its pooled output, which
    # was a real contributor to the original version of this project
    # occasionally steering the wrong direction).
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
    print(f"\nDone. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        img_dir, labels_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        img_dir = Path(__file__).parent / "data" / "images"
        labels_csv = Path(__file__).parent / "data" / "labels.csv"
        print(f"No arguments given — using defaults:\n  images_dir = {img_dir}\n  labels_csv = {labels_csv}\n")
    else:
        raise SystemExit("Usage: python train.py <images_dir> <labels_csv>")
    main(img_dir, labels_csv)
