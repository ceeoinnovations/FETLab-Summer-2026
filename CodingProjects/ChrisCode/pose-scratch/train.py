"""
Step 2 — Train the from-scratch CNN.

Every parameter starts from random noise — nothing is pretrained.
The network must learn to detect edges, shapes, and poses entirely
from the images you collected, which is why you need more of them.

Key differences from pose-pytorch:
  - No frozen layers (all weights update every step)
  - AdamW optimizer with weight decay to resist overfitting
  - Cosine LR schedule (gradually reduces learning rate)
  - More aggressive augmentation
  - 40 epochs instead of 20

Watch for overfitting: if train loss keeps dropping but val loss
rises, the model is memorizing rather than generalizing. Collect more
diverse images and re-run.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import build_model
import matplotlib.pyplot as plt

DATA_DIR  = "data"
MODEL_OUT = "pose_model.pt"
EPOCHS    = 40
BATCH     = 16
LR        = 5e-4

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
classes = dataset.classes
print(f"Classes found: {classes}")
print(f"Total images : {len(dataset)}")

if len(dataset) < len(classes) * 100:
    print(f"\n⚠  Fewer than 100 images per class on average.")
    print(f"   From-scratch training works best with 100+ per class.")
    print(f"   Results may be poor — consider collecting more data first.\n")

n_val   = max(1, int(0.2 * len(dataset)))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

device = ("mps"  if torch.backends.mps.is_available()  else
          "cuda" if torch.cuda.is_available()           else "cpu")
print(f"Training on : {device}\n")

model = build_model(num_classes=len(classes)).to(device)

# All weights train from scratch — nothing is frozen
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses, val_accs = [], [], []
best_val_loss = float("inf")

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
    scheduler.step()

    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out      = model(imgs)
            val_loss += criterion(out, labels).item()
            correct  += (out.argmax(1) == labels).sum().item()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    acc       = correct    / len(val_ds)
    lr_now    = scheduler.get_last_lr()[0]
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    val_accs.append(acc)
    print(f"Epoch {epoch + 1:2d}/{EPOCHS}  "
          f"train={avg_train:.4f}  val={avg_val:.4f}  acc={acc:.1%}  lr={lr_now:.2e}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({"state_dict": model.state_dict(), "classes": classes}, MODEL_OUT)
        print(f"           ↑ best model saved → {MODEL_OUT}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(train_losses, label="train")
ax1.plot(val_losses,   label="val")
ax1.set(title="Loss (from scratch)", xlabel="Epoch", ylabel="Cross-entropy")
ax1.legend()

ax2.plot(val_accs, color="green")
ax2.set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to {MODEL_OUT}  |  Plot saved to training_curve.png")
