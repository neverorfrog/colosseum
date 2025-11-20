from mjlab.tasks.registry import register_mjlab_task

register_mjlab_task(
    task_id="Mjlab-Cartpole",
    env_cfg=CARTPOLE_ENV_CFG,
    rl_cfg=CARTPOLE_RL_CFG,
)