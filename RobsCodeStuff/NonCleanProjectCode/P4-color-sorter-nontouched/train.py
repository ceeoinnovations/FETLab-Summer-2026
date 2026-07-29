"""
Step 2 — Train a K-Nearest Neighbors classifier on the collected readings.

KNN is the simplest possible ML classifier: to classify a new reading,
find the K training samples closest to it in feature space and take a
majority vote. No gradient descent, no loss function — just distance.

Distance is computed after standardizing all features (zero mean, unit
variance) so that a large-range feature like rawRed doesn't dominate
a small-range one like reflection.

Saves the trained pipeline (scaler + KNN) to color_model.pkl.
"""

import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
from config import FEATURES, K

HERE = Path(__file__).parent

DATA_FILE  = HERE / "color_data.csv"
MODEL_FILE = HERE / "color_model.pkl"

# ── Load data ─────────────────────────────────────────────────────────────────
labels, rows = [], []
with open(DATA_FILE) as f:
    for row in csv.DictReader(f):
        labels.append(row["label"])
        rows.append([float(row[feat]) for feat in FEATURES])

X = np.array(rows)
y = np.array(labels)
classes = sorted(set(y))

print(f"Loaded {len(X)} samples across {len(classes)} classes: {classes}")
for cls in classes:
    print(f"  {cls}: {(y == cls).sum()} samples")

# ── Train ─────────────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),      # normalize features to same scale
    ("knn",    KNeighborsClassifier(n_neighbors=K, metric="euclidean")),
])

# Cross-validation gives an honest accuracy estimate before we commit
cv      = StratifiedKFold(n_splits=min(5, min((y == c).sum() for c in classes)))
cv_accs = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
print(f"\n{cv.n_splits}-fold cross-validation accuracy: "
      f"{cv_accs.mean():.1%} ± {cv_accs.std():.1%}")

# Fit on all data for the final model
pipeline.fit(X, y)
joblib.dump({"pipeline": pipeline, "classes": classes, "features": FEATURES}, MODEL_FILE)
print(f"Model saved → {MODEL_FILE}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

train_preds = pipeline.predict(X)
ConfusionMatrixDisplay.from_predictions(
    y, train_preds, display_labels=classes, ax=axes[0], colorbar=False
)
axes[0].set_title("Confusion matrix (training data)")

# ── 3-D RGB scatter ───────────────────────────────────────────────────────────
# Plot raw R, G, B in 3-D space — the physical color of each point
# should roughly match its position, which is a great sanity check.
r_idx = FEATURES.index("rawRed")
g_idx = FEATURES.index("rawGreen")
b_idx = FEATURES.index("rawBlue")

ax3d = fig.add_subplot(1, 2, 2, projection="3d")

# Map class names to approximate display colors for the scatter
DISPLAY_COLORS = {
    "red": "#e74c3c", "blue": "#3498db", "green": "#2ecc71",
    "yellow": "#f1c40f", "white": "#ecf0f1", "black": "#2c3e50",
    "orange": "#e67e22", "purple": "#9b59b6", "pink": "#ff69b4",
}
for cls in classes:
    mask = y == cls
    ax3d.scatter(
        X[mask, r_idx], X[mask, g_idx], X[mask, b_idx],
        label=cls,
        color=DISPLAY_COLORS.get(cls, "gray"),
        edgecolors="black", linewidths=0.4, s=60, alpha=0.85,
    )
ax3d.set_xlabel("rawRed (norm)")
ax3d.set_ylabel("rawGreen (norm)")
ax3d.set_zlabel("rawBlue (norm)")
ax3d.set_title("RGB feature space")
ax3d.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plot_out = HERE / "training_results.png"
plt.savefig(plot_out, dpi=150)
plt.show()
print(f"Plot saved → {plot_out}")
