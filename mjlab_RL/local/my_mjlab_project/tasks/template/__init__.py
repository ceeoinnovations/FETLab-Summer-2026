"""
TEMPLATE: PPO runner config + task registration.

This part of the pattern barely varies between tasks 
(scale hidden_dims down for small/low-dimensional tasks, e.g. a
2-joint crawler with ~10 obs dims used (64, 64); scale up for higher-
dimensional tasks, e.g. a 6-DOF arm with ~18 obs dims used (128, 128)).
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task

from .example_env_cfg import get_env_cfg  # TODO: match your actual env_cfg filename/function


def ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(64, 64),  # TODO: scale with obs/action dimensionality
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(64, 64),  # TODO: match actor, or size up if critic obs group is much richer
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,  # raise if training collapses to a low-effort
                                 # local optimum before ever exploring past it
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="TODO_experiment_name",  # used for the logs/ folder path
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=1_000,  # TODO: scale with task difficulty
    )


register_mjlab_task(
    task_id="Mjlab-TODO-Task-Name",  # this exact string is what `uv run play`/train takes as an argument
    env_cfg=get_env_cfg(),
    play_env_cfg=get_env_cfg(play=True),
    rl_cfg=ppo_runner_cfg(),
)

# TODO: if you want variants of the same task (e.g. with/without some
# reward term, or a different action scheme), register additional
# task_ids here -- but each task_id string must be UNIQUE across your
# entire project. We hit a real bug from exactly this: two separate
# files in the same project both registered "Mjlab-UR3e-Reach", and the
# second one silently broke the ENTIRE package import (not just that one
# task) with `ValueError: Task 'Mjlab-UR3e-Reach' is already registered`,
# because Python aborts a module's execution on the first uncaught
# exception -- anything after that failed import in your project's root
# __init__.py never runs, so even unrelated tasks disappear from the
# CLI's choices until the duplicate is fixed.