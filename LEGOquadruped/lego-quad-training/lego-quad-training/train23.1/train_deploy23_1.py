"""Train the DEPLOYMENT policy on the FAITHFUL MESH MODEL, v23.1 (5 Hz).

    uv run python train_deploy23_1.py

v23.1 = train23 (surgical anti-front-sync retrain of 17.1) + RECOVERY training.
train23 PREVENTS the front pair from syncing (forward-gated diagonal coordination
+ back-active balance); v23.1 additionally teaches it to RE-separate a pair that
has already synced:
  - BROKEN-phase RSI seeds: half the seeded episodes start with one pair co-
    rotating (the synced/bad state) so the policy must break out of it.
  - Mid-episode phase kicks: occasional crank-velocity perturbations force it to
    re-coordinate a disturbed gait, not just sustain a clean one.
  - RSI_PROB lowered 0.85 -> 0.6 (more varied / cold starts).
Keeps train23's forward-gated coordination (barrier_adv_coef=1.5) and the back-
active balance, so unlike train19.1 it should NOT collapse to front-only.
Produces lego_quad_deploy_ppo23_1.zip.
"""
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from lego_env_deploy23_1 import LegoQuadDeployEnv
from two_critic_ppo import TwoCriticPPO, TwoCriticPolicy

N_ENVS = 48
TOTAL_STEPS = 3_000_000
MODEL_PATH = Path("lego_quad_deploy_ppo23_1.zip")
BARRIER_ADV_COEF = 1.5   # same as train23 (forward-gated diagonal coordination)

IMU_MODE = "live"
EPISODE_S = 30.0


def make_env():
  return LegoQuadDeployEnv(imu_mode=IMU_MODE, episode_s=EPISODE_S)


def main():
  env = VecMonitor(SubprocVecEnv([make_env for _ in range(N_ENVS)]))
  if MODEL_PATH.exists():
    model = TwoCriticPPO.load(MODEL_PATH, env=env, device="cpu")
  else:
    model = TwoCriticPPO(TwoCriticPolicy, env, device="cpu", n_steps=256,
                         batch_size=1024, learning_rate=3e-4, ent_coef=0.005,
                         barrier_adv_coef=BARRIER_ADV_COEF,
                         policy_kwargs=dict(net_arch=[256, 128, 64]), verbose=1)
  ckpt = CheckpointCallback(save_freq=max(100_000 // N_ENVS, 1),
                            save_path="./checkpoints", name_prefix="deploy23_1")
  model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt, progress_bar=True)
  model.save(MODEL_PATH)
  print(f"saved {MODEL_PATH} - now run: python export_policy.py")


if __name__ == "__main__":
  main()
