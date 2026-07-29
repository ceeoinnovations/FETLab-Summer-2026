"""
Train the grid detector (grid_detector_model.py) on pseudo-labeled boxes
from generate_pseudo_boxes.py.

    python generate_pseudo_boxes.py data/images data/pseudo_boxes.csv
    python train_grid_detector.py data/images data/pseudo_boxes.csv

Loss has two parts, added together:
  - Confidence: binary cross-entropy over EVERY cell in the grid (49 of
    them for a 7x7 grid) — the network needs to learn what "nothing here"
    looks like just as much as what "object here" looks like.
  - Box (x_offset, y_offset, w, h): squared error, but ONLY at the single
    cell responsible for the real object — the other 48 cells have no
    real box to be right or wrong about, so they're excluded from this
    part of the loss entirely (matches how real YOLO's loss works).

Uses the same session-aware train/val split reasoning as train.py and
train_detector.py: splits by whole drive if collect_data.py's session_id
column is available, since consecutive frames are otherwise too similar
for a random per-frame split to give an honest validation estimate.
"""

import sys
import csv
import random
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from grid_detector_model import GridDetector, GRID_SIZE
from config import IMG_SIZE

MODEL_OUT = "grid_detector_model.pt"
EPOCHS = 15
BATCH = 24
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
UNFREEZE_LAST_N_BLOCKS = 2
VAL_FRACTION = 0.2
SEED = 42
BOX_LOSS_WEIGHT = 5.0  # box accuracy matters more than the many easy "empty cell" predictions


class GridDataset(Dataset):
    def __init__(self, img_dir, rows, transform):
        self.img_dir = Path(img_dir)
        self.rows = rows  # (fname, visible, row, col, x_off, y_off, w, h)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fname, visible, row, col, x_off, y_off, w, h = self.rows[idx]
        img = Image.open(self.img_dir / fname).convert("RGB")
        img = self.transform(img)

        conf_target = torch.zeros(GRID_SIZE, GRID_SIZE)
        box_target = torch.zeros(4, GRID_SIZE, GRID_SIZE)
        box_mask = torch.zeros(GRID_SIZE, GRID_SIZE)

        if int(visible) == 1:
            r, c = int(row), int(col)
            conf_target[r, c] = 1.0
            box_target[0, r, c] = float(x_off)
            box_target[1, r, c] = float(y_off)
            box_target[2, r, c] = float(w)
            box_target[3, r, c] = float(h)
            box_mask[r, c] = 1.0

        return img, conf_target, box_target, box_mask


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def session_aware_split(img_dir, labels_csv):
    with open(labels_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    sibling_labels = Path(labels_csv).parent / "labels.csv"
    session_by_fname = {}
    if sibling_labels.exists():
        with open(sibling_labels) as f:
            r = csv.reader(f)
            h = next(r)
            if "session_id" in h:
                sid_idx = h.index("session_id")
                for row in r:
                    session_by_fname[row[0]] = row[sid_idx]

    rows_by_session = defaultdict(list)
    for row in rows:
        sid = session_by_fname.get(row[0], "0")
        rows_by_session[sid].append(tuple(row))

    if not session_by_fname:
        print("NOTE: no matching labels.csv with session_id found — falling "
              "back to a random per-frame split (val loss will be a bit "
              "optimistic).")

    session_ids = list(rows_by_session.keys())
    rng = random.Random(SEED)
    rng.shuffle(session_ids)

    if len(session_ids) < 2:
        only_rows = next(iter(rows_by_session.values()))
        idx = list(range(len(only_rows)))
        rng.shuffle(idx)
        n_val = max(1, int(VAL_FRACTION * len(idx)))
        val_idx, train_idx = set(idx[:n_val]), set(idx[n_val:])
        return [only_rows[i] for i in train_idx], [only_rows[i] for i in val_idx]

    n_val_sessions = max(1, int(VAL_FRACTION * len(session_ids)))
    n_val_sessions = min(n_val_sessions, len(session_ids) - 1)
    val_sessions = set(session_ids[:n_val_sessions])
    train_rows, val_rows = [], []
    for sid, sid_rows in rows_by_session.items():
        (val_rows if sid in val_sessions else train_rows).extend(sid_rows)
    return train_rows, val_rows


def compute_loss(model_out, conf_target, box_target, box_mask, bce):
    confidence, x_offset, y_offset, w, h = model_out
    pred_box = torch.stack([x_offset, y_offset, w, h], dim=1)  # (B, 4, S, S)

    conf_loss = bce(confidence, conf_target)

    n_responsible = box_mask.sum().clamp(min=1)
    box_err = ((pred_box - box_target) ** 2) * box_mask.unsqueeze(1)
    box_loss = box_err.sum() / (n_responsible * 4)

    return conf_loss + BOX_LOSS_WEIGHT * box_loss


def main(img_dir, labels_csv):
    train_rows, val_rows = session_aware_split(img_dir, labels_csv)
    print(f"Frames: {len(train_rows)} train / {len(val_rows)} val")

    train_ds = GridDataset(img_dir, train_rows, transform)
    val_ds = GridDataset(img_dir, val_rows, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    model = GridDetector().to(device)
    for p in model.features.parameters():
        p.requires_grad = False
    backbone_finetune_params = []
    for block in list(model.features.children())[-UNFREEZE_LAST_N_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True
            backbone_finetune_params.append(p)

    optimizer = torch.optim.Adam([
        {"params": model.head.parameters(), "lr": LR_HEAD},
        {"params": backbone_finetune_params, "lr": LR_BACKBONE},
    ])
    bce = nn.BCELoss()

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, conf_t, box_t, mask_t in train_loader:
            imgs, conf_t, box_t, mask_t = (imgs.to(device), conf_t.to(device),
                                            box_t.to(device), mask_t.to(device))
            optimizer.zero_grad()
            out = model(imgs)
            loss = compute_loss(out, conf_t, box_t, mask_t, bce)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, conf_t, box_t, mask_t in val_loader:
                imgs, conf_t, box_t, mask_t = (imgs.to(device), conf_t.to(device),
                                                box_t.to(device), mask_t.to(device))
                out = model(imgs)
                val_loss += compute_loss(out, conf_t, box_t, mask_t, bce).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1:2d}/{EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({"state_dict": model.state_dict()}, MODEL_OUT)
            print(f"           ↑ best model saved → {MODEL_OUT}")

    print(f"\nDone. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        img_dir, labels_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        img_dir = Path(__file__).parent / "data" / "images"
        labels_csv = Path(__file__).parent / "data" / "pseudo_boxes.csv"
        print(f"No arguments given — using defaults:\n  images_dir = {img_dir}\n  labels_csv = {labels_csv}\n")
    else:
        raise SystemExit("Usage: python train_grid_detector.py <images_dir> <pseudo_boxes_csv>")
    main(img_dir, labels_csv)
