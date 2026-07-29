"""
gesture_classifier.py

A small nearest-neighbor ("k-NN") classifier over the hand-landmark
feature vectors saved by gesture_data_collector.py.

This mirrors the "store every example, compare new readings to all of
them" approach from the color-sensor unsupervised learning activity:
no training step is required -- classification just measures distance
from a new example to every saved example and takes a majority vote
among the closest matches.
"""

import json
import numpy as np


class GestureClassifier:
    def __init__(self, data_file="gesture_data.json", k=5):
        self.k = k
        self.labels = []
        self.vectors = None
        self._load(data_file)

    def _load(self, data_file):
        with open(data_file, "r") as f:
            data = json.load(f)

        labels = []
        vectors = []
        for label, examples in data.items():
            for example in examples:
                labels.append(label)
                vectors.append(example)

        if not vectors:
            raise ValueError(
                f"No training examples found in {data_file}. "
                "Run gesture_data_collector.py first and record some examples."
            )

        self.labels = labels
        self.vectors = np.array(vectors, dtype=np.float32)

    def predict(self, feature_vector):
        """
        Returns (predicted_label, confidence) where confidence is the
        fraction of the k nearest neighbors that agreed on the label
        (1.0 = all k neighbors agreed, 0.2 = only 1/5 agreed).
        """
        distances = np.linalg.norm(self.vectors - feature_vector, axis=1)
        k = min(self.k, len(distances))
        nearest_indices = np.argpartition(distances, k - 1)[:k]
        nearest_labels = [self.labels[i] for i in nearest_indices]

        counts = {}
        for label in nearest_labels:
            counts[label] = counts.get(label, 0) + 1

        best_label = max(counts, key=counts.get)
        confidence = counts[best_label] / k
        return best_label, confidence