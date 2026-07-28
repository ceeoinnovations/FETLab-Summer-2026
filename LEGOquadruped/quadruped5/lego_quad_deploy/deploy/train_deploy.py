"""Train the DEPLOYMENT policy (hardware-matched observations, 20 Hz).

    uv run python train_deploy.py

Produces lego_quad_deploy_ppo.zip. Run export_policy.py afterwards.
"""
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from lego_env_deploy2 import LegoQuadDeployEnv

N_ENVS = 8
TOTAL_STEPS = 3_000_000
MODEL_PATH = Path("lego_quad_deploy_ppo3.zip")


def make_env():
  return LegoQuadDeployEnv()


def main():
  env = VecMonitor(SubprocVecEnv([make_env for _ in range(N_ENVS)]))
  if MODEL_PATH.exists():
    model = PPO.load(MODEL_PATH, env=env, device="cpu")
  else:
    model = PPO("MlpPolicy", env, device="cpu", n_steps=512, batch_size=1024,
                learning_rate=3e-4, ent_coef=0.005,
                policy_kwargs=dict(net_arch=[256, 128, 64]), verbose=1)
  ckpt = CheckpointCallback(save_freq=max(100_000 // N_ENVS, 1),
                            save_path="./checkpoints", name_prefix="deploy")
  model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt, progress_bar=True)
  model.save(MODEL_PATH)
  print(f"saved {MODEL_PATH} - now run: python export_policy.py")


if __name__ == "__main__":
  main()
