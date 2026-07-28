"""
Check whether collected data actually has what offline RL needs, BEFORE
spending time on train_offline_rl.py.

Background: offline RL can only learn "this action was better than that
one" by seeing the SAME visual situation handled with genuinely DIFFERENT
actions, with different outcomes. During development, autonomous data from
deterministic controllers (hardcoded or another trained model driving
itself) came back with within-situation action variance at ~85-90% of the
dataset's overall variance — essentially no usable contrast, no matter how
"messy" the driving looked. Real human joystick data came back at ~35-40%
on the same check. This script runs that exact diagnostic on whatever data
you just collected, so a bad dataset gets caught here instead of after a
training run that looks fine on paper but produces a policy with nothing
real to learn from.

Usage:
    python check_data_diversity.py data/images data/pseudo_labels.csv
"""

import sys
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_data(labels_csv, pseudo_csv):
    with open(labels_csv) as f:
        labels = list(csv.DictReader(f))
    with open(pseudo_csv) as f:
        pseudo = list(csv.DictReader(f))
    if len(labels) != len(pseudo):
        raise SystemExit(f"Row count mismatch: {labels_csv} has {len(labels)} rows, "
                          f"{pseudo_csv} has {len(pseudo)}. Did you regenerate pseudo-labels "
                          "after collecting more data?")
    return labels, pseudo


def diversity_ratio(states, actions, min_bucket_n=8):
    """Within-visual-situation action std, divided by overall action std.
    Lower = more real contrast for offline RL to learn from. ~0.85-0.90 was
    what deterministic-controller data measured at (essentially useless);
    ~0.35-0.40 was what good human data measured at."""
    cx, area = states[:, 0], states[:, 2]
    cx_bins = np.digitize(cx, np.linspace(-1, 1, 9))
    area_bins = np.digitize(area, np.linspace(0, 0.3, 5))
    buckets = defaultdict(list)
    for i in range(len(states)):
        buckets[(cx_bins[i], area_bins[i])].append(actions[i])
    spreads = [np.array(v).std(axis=0).mean() for v in buckets.values() if len(v) >= min_bucket_n]
    n_qualify = sum(1 for v in buckets.values() if len(v) >= min_bucket_n)
    if not spreads:
        return None, len(buckets), n_qualify
    within = np.mean(spreads)
    overall = actions.std(axis=0).mean()
    return (within / overall if overall > 0 else None), len(buckets), n_qualify


def main(labels_csv, pseudo_csv):
    labels, pseudo = load_data(labels_csv, pseudo_csv)

    states = np.array([[float(p["cx_norm"]), float(p["cy_norm"]),
                         float(p["area_frac"]), float(p["visible"])] for p in pseudo], dtype=np.float32)
    actions = np.array([[float(l["left_speed"]), float(l["right_speed"])]
                        for l in labels], dtype=np.float32) / 100.0

    sessions = set(l["session_id"] for l in labels)
    n_visible = int(states[:, 3].sum())

    print(f"{len(labels)} frames, {len(sessions)} session(s), "
          f"{n_visible}/{len(labels)} ({100*n_visible/len(labels):.1f}%) visible\n")

    ratio, n_buckets, n_qualify = diversity_ratio(states, actions)
    if ratio is None:
        print("Not enough data to compute a meaningful diversity ratio yet "
              "(need more frames, or more visual variety).")
        return

    print(f"Diversity ratio: {ratio:.2f}  ({n_qualify}/{n_buckets} visual situations "
          f"had enough samples to check)")
    print()
    if ratio >= 0.75:
        print("VERDICT: Low diversity — close to what deterministic-controller data")
        print("measured at (~0.85-0.90). This data likely lacks the same-situation-")
        print("different-action contrast offline RL needs. If this came from an")
        print("autonomous drive.py run rather than a human at the joystick, that's")
        print("almost certainly why — see config.py's note on this. Training on it")
        print("anyway will probably just reproduce plain imitation, or worse,")
        print("training instability.")
    elif ratio >= 0.5:
        print("VERDICT: Middling diversity. Better than deterministic-controller data,")
        print("but not as strong a signal as the human data that worked well during")
        print("development (~0.35-0.40). Worth training on, but don't be surprised if")
        print("the result is close to plain imitation rather than a clear improvement.")
    else:
        print("VERDICT: Good diversity — in the range that produced a stable, real")
        print("training signal during development. Reasonable to proceed to")
        print("train_offline_rl.py.")

    print("\nRemember: this ratio tells you whether there's local action variety, not")
    print("whether that variety actually correlates with better outcomes. If you want")
    print("that deeper check, look at how train_offline_rl.py's printed action-std")
    print("comparison behaves after training, not just this number beforehand.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        images_dir, pseudo_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        images_dir = Path(__file__).parent / "data" / "images"
        pseudo_csv = Path(__file__).parent / "data" / "pseudo_labels.csv"
        print(f"No arguments given — using defaults:\n  images_dir  = {images_dir}\n"
              f"  pseudo_csv  = {pseudo_csv}\n")
    else:
        raise SystemExit("Usage: python check_data_diversity.py <images_dir> <pseudo_labels_csv>")

    labels_csv = Path(images_dir).parent / "labels.csv"
    main(labels_csv, pseudo_csv)
