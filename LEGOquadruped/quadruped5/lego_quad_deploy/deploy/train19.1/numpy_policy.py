"""Pure-NumPy MLP inference for an exported SB3 PPO policy.

No torch needed - runs on any Python (including 3.14 for the LEGO API).
"""

import numpy as np


class NumpyPolicy:
  def __init__(self, npz_path):
    d = np.load(npz_path)
    self.layers = []
    i = 0
    while f"w{i}" in d:
      self.layers.append((d[f"w{i}"], d[f"b{i}"]))
      i += 1
    self.out_w, self.out_b = d["out_w"], d["out_b"]

  def __call__(self, obs):
    x = np.asarray(obs, dtype=np.float64)
    for w, b in self.layers:
      x = np.tanh(x @ w.T + b)
    a = x @ self.out_w.T + self.out_b   # deterministic action = mean
    return np.clip(a, -1.0, 1.0)
