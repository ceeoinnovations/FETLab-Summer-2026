"""
Train the classifier (classifier_model.py) on pseudo-labeled categories
from generate_pseudo_classes.py.

    python generate_pseudo_classes.py data/images data/pseudo_classes.csv
    python train_classifier.py data/images data/pseudo_classes.csv

Loss is ordinary cross-entropy — "pick the right one of N categories" —
the standard loss for classification, different from the squared-error
loss the other two options use for continuous coordinates/boxes.

Classes are very likely imbalanced (most driving is "straight", not
"hard_left"), so training samples are weighted inversely to their class's
frequency, similar in spirit to the turning-weighted sampler used for the
original motor-speed model, but computed per-category here rather than
per-continuous-value.

Uses the same session-aware train/val split reasoning as the other two
training scripts: splits by whole drive if collect_data.py's session_id
column is available.
"""

import sys
import csv
import random
import torch
import torch.nn as nn
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
from classifier_model import build_classifier_model
from config import IMG_SIZE, CATEGORIES

MODEL_OUT = "classifier_model.pt"
EPOCHS = 15
BATCH = 30
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
UNFREEZE_LAST_N_BLOCKS = 2
VAL_FRACTION = 0.2
SEED = 42

CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}


class ClassifierDataset(Dataset):
    def __init__(self, img_dir, rows, transform):
        self.img_dir = Path(img_dir)
        self.rows = rows  # (fname, category)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fname, category = self.rows[idx]
        img = Image.open(self.img_dir / fname).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(CATEGORY_TO_IDX[category], dtype=torch.long)
        return img, label


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
        rows = [tuple(row) for row in reader]

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
        rows_by_session[sid].append(row)

    if not session_by_fname:
        print("NOTE: no matching labels.csv with session_id found — falling "
              "back to a random per-frame split (val accuracy will be a bit "
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


def main(img_dir, labels_csv):
    train_rows, val_rows = session_aware_split(img_dir, labels_csv)
    print(f"Frames: {len(train_rows)} train / {len(val_rows)} val")

    train_counts = Counter(r[1] for r in train_rows)
    print("Train category counts:", dict(train_counts))
    missing = [c for c in CATEGORIES if train_counts.get(c, 0) == 0]
    if missing:
        raise SystemExit(f"No training examples at all for: {missing}. "
                          "Collect more varied data or adjust the bucket "
                          "thresholds in config.py before training.")

    train_ds = ClassifierDataset(img_dir, train_rows, transform)
    val_ds = ClassifierDataset(img_dir, val_rows, transform)

    # Weight samples inversely to their category's frequency, so the
    # rarer categories (e.g. hard_left) aren't drowned out by "straight".
    weight_per_category = {c: 1.0 / train_counts[c] for c in CATEGORIES}
    sample_weights = [weight_per_category[r[1]] for r in train_rows]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    model = build_classifier_model().to(device)
    for p in model.features.parameters():
        p.requires_grad = False
    backbone_finetune_params = []
    for block in list(model.features.children())[-UNFREEZE_LAST_N_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True
            backbone_finetune_params.append(p)

    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
        {"params": backbone_finetune_params, "lr": LR_BACKBONE},
    ])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                pred = out.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total else 0.0
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1:2d}/{EPOCHS}  train_loss={avg_train:.4f}  "
              f"val_loss={avg_val:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict()}, MODEL_OUT)
            print(f"           ↑ best model saved → {MODEL_OUT}")

    print(f"\nDone. Best validation accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        img_dir, labels_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        img_dir = Path(__file__).parent / "data" / "images"
        labels_csv = Path(__file__).parent / "data" / "pseudo_classes.csv"
        print(f"No arguments given — using defaults:\n  images_dir = {img_dir}\n  labels_csv = {labels_csv}\n")
    else:
        raise SystemExit("Usage: python train_classifier.py <images_dir> <pseudo_classes_csv>")
    main(img_dir, labels_csv)
