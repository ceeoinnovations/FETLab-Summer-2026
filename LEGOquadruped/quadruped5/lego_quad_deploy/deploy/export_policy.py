"""Export an SB3 PPO policy to NumPy weights + verify the match.

    uv run python export_policy.py lego_quad_deploy_ppo.zip

Writes policy_weights6.npz next to it.
"""
import sys

import numpy as np
from stable_baselines3 import PPO

from numpy_policy import NumpyPolicy

path = sys.argv[1] if len(sys.argv) > 1 else "lego_quad_deploy_ppo6.zip"
model = PPO.load(path, device="cpu")

params = {}
pi = model.policy.mlp_extractor.policy_net  # hidden layers (Linear+Tanh)
linears = [m for m in pi if m.__class__.__name__ == "Linear"]
for i, lin in enumerate(linears):
  params[f"w{i}"] = lin.weight.detach().numpy()
  params[f"b{i}"] = lin.bias.detach().numpy()
params["out_w"] = model.policy.action_net.weight.detach().numpy()
params["out_b"] = model.policy.action_net.bias.detach().numpy()
np.save if False else np.savez("policy_weights6.npz", **params)

# verify NumPy inference matches SB3 exactly
np_policy = NumpyPolicy("policy_weights6.npz")
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
  obs = rng.normal(0, 1, model.observation_space.shape).astype(np.float32)
  a_sb3, _ = model.predict(obs, deterministic=True)
  a_np = np_policy(obs)
  worst = max(worst, float(np.max(np.abs(a_sb3 - a_np))))
print(f"exported policy_weights5.npz | max |SB3 - NumPy| over 200 random obs: {worst:.2e}")
assert worst < 1e-4, "mismatch - check architecture"
print("verified: NumPy inference matches SB3")
