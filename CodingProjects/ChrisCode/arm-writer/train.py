"""
Step 1 — Train the reaching policy with SAC.

Uses Stable-Baselines3's Soft Actor-Critic (SAC) algorithm, which is
off-policy and sample-efficient — ideal for continuous robotic control.

The policy learns to move the LED to any target position in the writing
canvas by receiving dense reward (-distance) and a success bonus.

Saves:
  arm_policy/        — trained SB3 model (load with SAC.load)
  training_log.npz   — episode rewards and lengths for visualize.py
"""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from env import ArmEnv
from config import (TOTAL_TIMESTEPS, N_ENVS, LEARNING_RATE, BUFFER_SIZE,
                    LEARNING_STARTS, BATCH_SIZE, TAU, GAMMA, POLICY_KWARGS)


class LogCallback(BaseCallback):
    """Collects per-episode mean rewards for the training plot."""
    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self.ep_lengths = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
        return True


# ── Environments ──────────────────────────────────────────────────────────────
print(f"Creating {N_ENVS} parallel environments...")
train_env = make_vec_env(ArmEnv, n_envs=N_ENVS)
eval_env  = ArmEnv()

# ── Model ─────────────────────────────────────────────────────────────────────
model = SAC(
    policy        = "MlpPolicy",
    env           = train_env,
    learning_rate = LEARNING_RATE,
    buffer_size   = BUFFER_SIZE,
    learning_starts = LEARNING_STARTS,
    batch_size    = BATCH_SIZE,
    tau           = TAU,
    gamma         = GAMMA,
    policy_kwargs = POLICY_KWARGS,
    verbose       = 1,
    tensorboard_log = "./tb_logs/",
)

print(f"\nPolicy network: {POLICY_KWARGS['net_arch']}")
print(f"Training for {TOTAL_TIMESTEPS:,} steps across {N_ENVS} envs...\n")

# ── Train ─────────────────────────────────────────────────────────────────────
log_cb  = LogCallback()
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = "./arm_policy_best/",
    log_path             = "./eval_logs/",
    eval_freq            = max(10_000 // N_ENVS, 1),
    n_eval_episodes      = 10,
    deterministic        = True,
    verbose              = 1,
)

model.learn(
    total_timesteps = TOTAL_TIMESTEPS,
    callback        = [log_cb, eval_cb],
    progress_bar    = True,
)

model.save("arm_policy")
print("\nModel saved → arm_policy.zip")

np.savez(
    "training_log.npz",
    ep_rewards = np.array(log_cb.ep_rewards),
    ep_lengths = np.array(log_cb.ep_lengths),
)
print("Training log saved → training_log.npz")
print("Run: python plan.py  (then visualize.py)")
