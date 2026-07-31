"""
PPO runner config + task registration for the lego crawler task.
Small network -- roughly 7 actor obs dims (2 joint pos + 2 joint vel + 3
projected gravity + 1 yaw + 2 last_action = wait, see crawler_env_cfg.py
for the exact obs group for a precise count) and 2 action dims.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task

from .crawler_env_cfg import get_crawler_env_cfg


def crawler_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="lego_crawler",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=3_000,
    )


register_mjlab_task(
    task_id="Mjlab-Lego-Crawler",
    env_cfg=get_crawler_env_cfg(),
    play_env_cfg=get_crawler_env_cfg(play=True),
    rl_cfg=crawler_ppo_runner_cfg(),
)