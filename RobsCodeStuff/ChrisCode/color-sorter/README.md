# color-sorter

A LEGO color sorter that ignores the sensor's built-in color classifier and builds its own using K-Nearest Neighbors on raw sensor measurements. You collect labeled readings, train a KNN model in seconds, and the robot sorts objects by color in real time.

---

## Why not use the built-in color detection?

The LEGO color sensor's internal classifier is tuned for the specific LEGO brick palette. It returns a single integer (Red, Blue, Green…) with no confidence, no adjustability, and no way to add new colors. Our approach:

- Uses the **raw 16-bit R, G, B** channels plus **reflection**, **hue**, **saturation**, and **value** — 7 features per reading
- Trains on **your objects under your lighting** — far more accurate for non-LEGO colors
- Gives a **confidence score** so the robot stops rather than guesses
- Takes **seconds to train** — no GPU, no neural network

---

## How it works

```
collect.py  →  train.py  →  sort.py
 (label)       (learn)      (sort)
```

1. **Collect** — Hold each colored object in front of the sensor, press a number key, and the 7 normalized sensor values are saved to `color_data.csv`.
2. **Train** — A KNN classifier is fit on those readings. Cross-validation reports honest accuracy. A confusion matrix and 3-D RGB scatter plot are saved.
3. **Sort** — The sensor reads continuously. Each reading is classified, and the double motor routes the object to the matching bin.

---

## The algorithm: K-Nearest Neighbors

KNN is about as simple as machine learning gets. To classify a new sensor reading **x**:

1. Compute the Euclidean distance from **x** to every training sample.
2. Find the **K** nearest neighbors.
3. The majority label among those K neighbors wins.
4. Confidence = fraction of neighbors that agree (e.g. 4 of 5 = 80%).

```
New reading:  R=0.42, G=0.12, B=0.08, reflection=0.71, …

Nearest neighbors:
  red   (dist=0.03)  ← 1
  red   (dist=0.05)  ← 2
  red   (dist=0.07)  ← 3
  orange(dist=0.11)  ← 4
  red   (dist=0.12)  ← 5

Vote: red=4, orange=1  →  predict "red"  confidence=80%
```

Before computing distances, all 7 features are **standardized** (zero mean, unit variance) so that a large-range feature like rawRed (0–65535 → 0–1) doesn't dwarf a small-range one like reflection (0–255 → 0–1).

---

## Sensor channels

The LEGO color sensor exposes far more than its built-in color name:

| Channel | Raw type | Normalized range | Description |
|---|---|---|---|
| `rawRed` | uint16 (0–65535) | 0–1 | Red photodiode response |
| `rawGreen` | uint16 (0–65535) | 0–1 | Green photodiode response |
| `rawBlue` | uint16 (0–65535) | 0–1 | Blue photodiode response |
| `reflection` | uint8 (0–255) | 0–1 | Total reflected light intensity |
| `hue` | uint16 (0–65535) | 0–1 | Color angle (red=0, green=~21845, blue=~43690) |
| `saturation` | uint8 (0–255) | 0–1 | Color purity (0=grey, 255=vivid) |
| `value` | uint8 (0–255) | 0–1 | Brightness |

All accessed via `cs.raw_reading()` in the extended `lelib.py`.

---

## Files

| File | Purpose |
|---|---|
| `lelib.py` | Extended SimpleLE — adds `raw_rgb()` and `raw_reading()` to `colorSensor` |
| `config.py` | Serial number, color→action map, features, K, confidence threshold |
| `collect.py` | Reads sensor, labels samples, appends to `color_data.csv` |
| `train.py` | Trains KNN, cross-validates, confusion matrix + 3-D RGB scatter |
| `sort.py` | Real-time classify + motor actuation |
| `requirements.txt` | Dependencies |

---

## Setup

### 1. Edit config.py

```python
SERIAL = 1128   # your LEGO Bluetooth card serial

COLOR_ACTIONS = {
    "red":   "right",    # red objects → right bin
    "blue":  "left",     # blue objects → left bin
    "green": "forward",  # green objects → forward bin
    "white": "stop",     # unknown → reject
}
```

Only include the colors you actually want to sort. You can add or remove entries freely.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 1: Collect readings

```bash
python collect.py
```

The terminal shows live sensor values updating in real time:
```
R=12341  G= 3421  B= 2891  refl=182  hue=  234
```

Hold a colored object steady in front of the sensor (≈2 cm away) and press its number key. Aim for **20–30 readings per color**.

**Tips:**
- Vary the distance slightly between readings (1–4 cm) — objects won't always be at the same distance.
- Vary the angle a few degrees — the sensor response shifts with angle.
- Collect under the same lighting you'll sort under. Bright sunlight vs. indoor fluorescent makes a big difference.
- You can run `collect.py` multiple times — new readings append to the CSV.

---

## Step 2: Train

```bash
python train.py
```

Output:
```
Loaded 120 samples across 3 classes: ['blue', 'green', 'red']
  blue:  42 samples
  green: 38 samples
  red:   40 samples

5-fold cross-validation accuracy: 97.5% ± 1.8%
Model saved → color_model.pkl
Plot saved  → training_results.png
```

**`training_results.png`** shows two panels:
- **Confusion matrix** — which colors get confused for which. A perfect classifier has only diagonal entries.
- **3-D RGB scatter** — each training sample plotted in (rawRed, rawGreen, rawBlue) space, colored by its label. If your colors cluster cleanly and don't overlap, the KNN will work well.

If accuracy is below ~90%: collect more samples, especially under the lighting conditions you'll actually sort in.

---

## Step 3: Sort

```bash
python sort.py
```

Output:
```
Color        Conf  Action      L      R
────────────────────────────────────────────────
red          100%  right     +100    -70
red          100%  right     +100    -70
blue?         40%  stop        +0     +0   ← low confidence → stop
blue          80%  left       -70   +100
```

Place colored objects in front of the sensor one at a time. The double motor actuates to the corresponding direction. Objects with low-confidence readings (marked with `?`) trigger a stop instead of a wrong sort.

**Adjust `CONFIDENCE_THRESHOLD`** in `config.py`:
- Raise it (e.g. `0.8`) if the robot makes wrong sorts
- Lower it (e.g. `0.4`) if it stops too often on valid objects

Press **Ctrl+C** to stop.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Low cross-validation accuracy | Collect more samples; ensure consistent lighting; check that objects are distinct colors |
| One color always predicted | Class imbalance — balance your sample counts per color |
| Confidence always low | Objects are too similar in RGB space; add more distinctive colors or collect more varied samples |
| Motor actuates briefly then stops | Normal behavior — reading happens at 10 Hz; the motor command updates each cycle |
| `color_model.pkl not found` | Run `train.py` first |
