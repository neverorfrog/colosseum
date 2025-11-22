from mjlab.tasks.registry import register_mjlab_task

from colosseum.train.tasks.cartpole.cartpole_env_cfg import (
    env_config,
    rl_config,
)

register_mjlab_task(
    task_id="Mjlab-Cartpole",
    env_cfg=env_config,
    play_env_cfg=env_config,
    rl_cfg=rl_config,
)
