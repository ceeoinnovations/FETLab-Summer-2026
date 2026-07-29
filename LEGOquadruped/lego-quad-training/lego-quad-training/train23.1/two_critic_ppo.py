"""Two-critic PPO for concept (2) of paper 2409.15780 (SB3 2.9.0).

Two independent value functions - one for R_standard (the multiplicative task
reward), one for R_barrier (the phase-clock contact barrier, delivered via
info["r_barrier"]). PPO's surrogate uses the SUM of the two advantages, each
NORMALIZED INDEPENDENTLY, and each critic is trained against its own returns.
This keeps the barrier's sharp penalty in its own value estimate and advantage
scale, so it can shape the gait hard without corrupting the task value (which
is what would otherwise make "fall to escape the penalty" look optimal).

Subclasses: ActorCriticPolicy (2nd value head), RolloutBuffer (2nd reward/value/
advantage/return stream), PPO (collect_rollouts + train).
"""
from typing import NamedTuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, obs_as_tensor


class TwoCriticSamples(NamedTuple):
  observations: th.Tensor
  actions: th.Tensor
  old_values: th.Tensor
  old_log_prob: th.Tensor
  advantages: th.Tensor
  returns: th.Tensor
  advantages2: th.Tensor
  returns2: th.Tensor
  old_values2: th.Tensor


class TwoCriticRolloutBuffer(RolloutBuffer):
  """RolloutBuffer with a second reward/value/advantage/return stream."""

  def reset(self):
    super().reset()
    shape = (self.buffer_size, self.n_envs)
    self.rewards2 = np.zeros(shape, dtype=np.float32)
    self.values2 = np.zeros(shape, dtype=np.float32)
    self.advantages2 = np.zeros(shape, dtype=np.float32)
    self.returns2 = np.zeros(shape, dtype=np.float32)

  def add(self, *args, reward2, value2, **kwargs):
    # store the 2nd stream at the current pos BEFORE super().add increments it
    self.rewards2[self.pos] = np.array(reward2)
    self.values2[self.pos] = value2.clone().cpu().numpy().flatten()
    super().add(*args, **kwargs)

  def compute_returns_and_advantage2(self, last_values2, dones):
    """GAE for the barrier stream (mirrors SB3's stream-1 computation)."""
    last_values2 = last_values2.clone().cpu().numpy().flatten()
    last_gae_lam = 0.0
    for step in reversed(range(self.buffer_size)):
      if step == self.buffer_size - 1:
        next_non_terminal = 1.0 - dones.astype(np.float32)
        next_values = last_values2
      else:
        next_non_terminal = 1.0 - self.episode_starts[step + 1]
        next_values = self.values2[step + 1]
      delta = (self.rewards2[step] + self.gamma * next_values * next_non_terminal
               - self.values2[step])
      last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
      self.advantages2[step] = last_gae_lam
    self.returns2 = self.advantages2 + self.values2

  def get(self, batch_size=None):
    assert self.full
    indices = np.random.permutation(self.buffer_size * self.n_envs)
    if not self.generator_ready:
      for tensor in ("observations", "actions", "values", "log_probs",
                     "advantages", "returns", "values2", "advantages2", "returns2"):
        self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
      self.generator_ready = True
    if batch_size is None:
      batch_size = self.buffer_size * self.n_envs
    start = 0
    total = self.buffer_size * self.n_envs
    while start < total:
      yield self._get_samples(indices[start:start + batch_size])
      start += batch_size

  def _get_samples(self, batch_inds, env=None):
    data = (
        self.observations[batch_inds],
        self.actions[batch_inds],
        self.values[batch_inds].flatten(),
        self.log_probs[batch_inds].flatten(),
        self.advantages[batch_inds].flatten(),
        self.returns[batch_inds].flatten(),
        self.advantages2[batch_inds].flatten(),
        self.returns2[batch_inds].flatten(),
        self.values2[batch_inds].flatten(),
    )
    return TwoCriticSamples(*tuple(map(self.to_torch, data)))


class TwoCriticPolicy(ActorCriticPolicy):
  """ActorCriticPolicy with a second value head (barrier critic)."""

  def _build(self, lr_schedule):
    super()._build(lr_schedule)
    self.value_net2 = nn.Linear(self.mlp_extractor.latent_dim_vf, 1).to(self.device)
    # rebuild optimizer so it includes value_net2's parameters
    self.optimizer = self.optimizer_class(
        self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

  def _latents(self, obs):
    features = self.extract_features(obs)
    if self.share_features_extractor:
      latent_pi, latent_vf = self.mlp_extractor(features)
    else:
      pi_features, vf_features = features
      latent_pi = self.mlp_extractor.forward_actor(pi_features)
      latent_vf = self.mlp_extractor.forward_critic(vf_features)
    return latent_pi, latent_vf

  def forward(self, obs, deterministic=False):
    latent_pi, latent_vf = self._latents(obs)
    values = self.value_net(latent_vf)
    values2 = self.value_net2(latent_vf)
    distribution = self._get_action_dist_from_latent(latent_pi)
    actions = distribution.get_actions(deterministic=deterministic)
    log_prob = distribution.log_prob(actions)
    actions = actions.reshape((-1, *self.action_space.shape))
    return actions, values, log_prob, values2

  def evaluate_actions(self, obs, actions):
    latent_pi, latent_vf = self._latents(obs)
    distribution = self._get_action_dist_from_latent(latent_pi)
    log_prob = distribution.log_prob(actions)
    values = self.value_net(latent_vf)
    values2 = self.value_net2(latent_vf)
    return values, log_prob, distribution.entropy(), values2

  def predict_values2(self, obs):
    features = super().extract_features(obs, self.vf_features_extractor)
    latent_vf = self.mlp_extractor.forward_critic(features)
    return self.value_net2(latent_vf)


class TwoCriticPPO(PPO):
  def __init__(self, policy=TwoCriticPolicy, env=None, barrier_adv_coef=1.0,
               **kwargs):
    # barrier_adv_coef scales the (independently-normalized) barrier advantage
    # relative to the task advantage in the surrogate. 1.0 = the original 1:1
    # blend; >1 leans harder on the gait/coordination critic, <1 lets the task
    # (forward + heading) critic dominate. This is the real gait-STRENGTH knob -
    # GAIT_WEIGHT washes out under per-stream normalization.
    self.barrier_adv_coef = barrier_adv_coef
    kwargs.setdefault("rollout_buffer_class", TwoCriticRolloutBuffer)
    super().__init__(policy, env, **kwargs)

  def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
    assert self._last_obs is not None
    self.policy.set_training_mode(False)
    n_steps = 0
    rollout_buffer.reset()
    if self.use_sde:
      self.policy.reset_noise(env.num_envs)
    callback.on_rollout_start()
    while n_steps < n_rollout_steps:
      if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
        self.policy.reset_noise(env.num_envs)
      with th.no_grad():
        obs_tensor = obs_as_tensor(self._last_obs, self.device)
        actions, values, log_probs, values2 = self.policy(obs_tensor)
      actions = actions.cpu().numpy()
      clipped_actions = actions
      if isinstance(self.action_space, spaces.Box):
        if self.policy.squash_output:
          clipped_actions = self.policy.unscale_action(clipped_actions)
        else:
          clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
      new_obs, rewards, dones, infos = env.step(clipped_actions)
      rewards2 = np.array([info.get("r_barrier", 0.0) for info in infos], dtype=np.float32)
      self.num_timesteps += env.num_envs
      callback.update_locals(locals())
      if not callback.on_step():
        return False
      self._update_info_buffer(infos, dones)
      n_steps += 1
      if isinstance(self.action_space, spaces.Discrete):
        actions = actions.reshape(-1, 1)
      for idx, done in enumerate(dones):
        if (done and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)):
          terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
          with th.no_grad():
            terminal_value = float(self.policy.predict_values(terminal_obs)[0])
            terminal_value2 = float(self.policy.predict_values2(terminal_obs)[0])
          rewards[idx] += self.gamma * terminal_value
          rewards2[idx] += self.gamma * terminal_value2
      rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts,
                         values, log_probs, reward2=rewards2, value2=values2)
      self._last_obs = new_obs
      self._last_episode_starts = dones
    with th.no_grad():
      last_obs_t = obs_as_tensor(new_obs, self.device)
      values = self.policy.predict_values(last_obs_t)
      values2 = self.policy.predict_values2(last_obs_t)
    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    rollout_buffer.compute_returns_and_advantage2(values2, dones)
    callback.update_locals(locals())
    callback.on_rollout_end()
    return True

  def train(self):
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    clip_range = self.clip_range(self._current_progress_remaining)
    clip_range_vf = (self.clip_range_vf(self._current_progress_remaining)
                     if self.clip_range_vf is not None else None)
    entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
    continue_training = True
    for epoch in range(self.n_epochs):
      approx_kl_divs = []
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
          actions = rollout_data.actions.long().flatten()
        values, log_prob, entropy, values2 = self.policy.evaluate_actions(
            rollout_data.observations, actions)
        values = values.flatten()
        values2 = values2.flatten()
        # concept (2): two advantages, each normalized independently, summed
        adv1, adv2 = rollout_data.advantages, rollout_data.advantages2
        if self.normalize_advantage and len(adv1) > 1:
          adv1 = (adv1 - adv1.mean()) / (adv1.std() + 1e-8)
          adv2 = (adv2 - adv2.mean()) / (adv2.std() + 1e-8)
        advantages = adv1 + getattr(self, "barrier_adv_coef", 1.0) * adv2
        ratio = th.exp(log_prob - rollout_data.old_log_prob)
        policy_loss = -th.min(advantages * ratio,
                              advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
        pg_losses.append(policy_loss.item())
        clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
        if clip_range_vf is None:
          values_pred, values_pred2 = values, values2
        else:
          values_pred = rollout_data.old_values + th.clamp(
              values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
          values_pred2 = rollout_data.old_values2 + th.clamp(
              values2 - rollout_data.old_values2, -clip_range_vf, clip_range_vf)
        value_loss = (F.mse_loss(rollout_data.returns, values_pred)
                      + F.mse_loss(rollout_data.returns2, values_pred2))
        value_losses.append(value_loss.item())
        entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
        entropy_losses.append(entropy_loss.item())
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
        with th.no_grad():
          log_ratio = log_prob - rollout_data.old_log_prob
          approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
          approx_kl_divs.append(approx_kl_div)
        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
          continue_training = False
          break
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
      self._n_updates += 1
      if not continue_training:
        break
    ev1 = explained_variance(self.rollout_buffer.values.flatten(),
                             self.rollout_buffer.returns.flatten())
    ev2 = explained_variance(self.rollout_buffer.values2.flatten(),
                             self.rollout_buffer.returns2.flatten())
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", ev1)
    self.logger.record("train/explained_variance2", ev2)
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)
