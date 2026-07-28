"""End-to-end smoke test for the two-critic PPO (concept 2).

Validates: env emits info["r_barrier"]; obs is 28-dim; and a short
TwoCriticPPO.learn() runs without error, exercising the 2nd advantage/value
stream and the truncation bootstrap for both streams.

    uv run python smoke_test10.py
"""
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from lego_env_deploy17 import LegoQuadDeployEnv
from two_critic_ppo import TwoCriticPPO, TwoCriticPolicy, TwoCriticRolloutBuffer

# 1) env contract
env0 = LegoQuadDeployEnv(imu_mode="live", episode_s=2.0)
obs, _ = env0.reset(seed=0)
assert obs.shape == (30,), obs.shape
_, r, term, trunc, info = env0.step(env0.action_space.sample())
assert "r_barrier" in info and info["r_barrier"] <= 1e-6, info
assert np.isfinite(r)
print(f"env OK: obs {obs.shape}, r_standard={r:+.3f}, r_barrier={info['r_barrier']:+.3f}")

# 2) short two-critic training run (2 envs, tiny rollout, episodes truncate at 10)
venv = VecMonitor(DummyVecEnv([lambda: LegoQuadDeployEnv(imu_mode="live", episode_s=2.0)
                               for _ in range(2)]))
model = TwoCriticPPO(TwoCriticPolicy, venv, device="cpu", n_steps=64, batch_size=64,
                     n_epochs=2, policy_kwargs=dict(net_arch=[32, 32]), verbose=0)
assert isinstance(model.rollout_buffer, TwoCriticRolloutBuffer)
assert hasattr(model.policy, "value_net2"), "second critic head missing"
model.learn(total_timesteps=256)
buf = model.rollout_buffer
assert np.isfinite(buf.returns2).all() and np.isfinite(buf.advantages2).all()
assert not np.allclose(buf.values2, 0.0), "critic 2 never produced nonzero values"
print(f"two-critic learn OK: {model.num_timesteps} steps, "
      f"critic2 returns range [{buf.returns2.min():+.3f},{buf.returns2.max():+.3f}]")
print("SMOKE OK")
