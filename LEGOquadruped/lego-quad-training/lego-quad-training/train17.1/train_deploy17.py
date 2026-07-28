"""Train the DEPLOYMENT policy on the FAITHFUL MESH MODEL, v17 (5 Hz).

    uv run python train_deploy17.py

v17: train15 (heading-aware, original direction) + two changes for a proper
FOUR-LEG trot that still steers, plus a sim-to-real fidelity fix:
  - FRONT-BACK balance reward (FB_WEIGHT): penalize front legs driving less than
    back, so all four contribute; leaves LEFT-RIGHT free so it can still steer
    (unlike train16's per-leg rotation floor, which killed steering).
  - GRIPPY RUBBER FEET: foot geoms get high friction (2.0, randomized 1.2-2.8),
    so a stationary foot grips the ground like the real rubber instead of
    sliding - closing the sim-to-real gap that let back-drive gaits cheat.
Keeps phase clock, two-critic, RSI, translation-dominant reward, heading-in-obs.
Produces lego_quad_deploy_ppo17.zip.
"""
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from lego_env_deploy17 import LegoQuadDeployEnv
from two_critic_ppo import TwoCriticPPO, TwoCriticPolicy

N_ENVS = 48   # use the exclusive node (was 16); ~3x faster, same learning
TOTAL_STEPS = 3_000_000
MODEL_PATH = Path("lego_quad_deploy_ppo17.zip")

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
                         policy_kwargs=dict(net_arch=[256, 128, 64]), verbose=1)
  ckpt = CheckpointCallback(save_freq=max(100_000 // N_ENVS, 1),
                            save_path="./checkpoints", name_prefix="deploy17")
  model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt, progress_bar=True)
  model.save(MODEL_PATH)
  print(f"saved {MODEL_PATH} - now run: python export_policy.py")


if __name__ == "__main__":
  main()
