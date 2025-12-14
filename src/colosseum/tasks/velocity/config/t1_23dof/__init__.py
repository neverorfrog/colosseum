"""Booster T1 velocity task configurations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import booster_t1_flat_env_cfg, booster_t1_rough_env_cfg
from .rl_cfg import booster_t1_ppo_runner_cfg

# Register T1 rough terrain velocity task
register_mjlab_task(
    task_id="Velocity-Rough-Booster-T1",
    env_cfg=booster_t1_rough_env_cfg(play=False),
    play_env_cfg=booster_t1_rough_env_cfg(play=True),
    rl_cfg=booster_t1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# Register T1 flat terrain velocity task
register_mjlab_task(
    task_id="Velocity-Flat-Booster-T1",
    env_cfg=booster_t1_flat_env_cfg(play=False),
    play_env_cfg=booster_t1_flat_env_cfg(play=True),
    rl_cfg=booster_t1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

__all__ = [
    "booster_t1_flat_env_cfg",
    "booster_t1_rough_env_cfg",
    "booster_t1_ppo_runner_cfg",
]
