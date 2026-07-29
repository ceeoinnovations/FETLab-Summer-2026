"""
Step 2 — Train the symbol classifier.

Uses MobileNetV2 pretrained on ImageNet with only the final layer
replaced and trained. The backbone already understands shapes and
edges, so 30-50 images per symbol is enough.

Saves the best model to symbol_model.pt and plots training_curve.png.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import build_model
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MODEL_OUT = "symbol_model.pt"
EPOCHS    = 30
BATCH     = 16
LR        = 1e-3

# Augmentations tuned for downward-facing camera on paper sketches:
#   - No horizontal flip (a left arrow flipped is a right arrow — bad)
#   - Random rotation to handle card placement angle
#   - Brightness/contrast jitter for lighting variation under the robot
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
classes = dataset.classes
print(f"Classes found: {classes}")
print(f"Total images : {len(dataset)}")

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

# Freeze backbone — only the new classification head trains
for p in model.features.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
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
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    val_accs.append(acc)
    print(f"Epoch {epoch + 1:2d}/{EPOCHS}  "
          f"train={avg_train:.4f}  val={avg_val:.4f}  acc={acc:.1%}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({"state_dict": model.state_dict(), "classes": classes}, MODEL_OUT)
        print(f"           ↑ best model saved → {MODEL_OUT}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(train_losses, label="train"); ax1.plot(val_losses, label="val")
ax1.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy"); ax1.legend()
ax2.plot(val_accs, color="green")
ax2.set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
ax2.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.show()
print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
