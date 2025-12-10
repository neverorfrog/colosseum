from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from .cartpole_scene import scene_config, simulation_config, viewer_config
from .cartpole_task import actions, events, observations, rewards, terminations

env_config = ManagerBasedRlEnvCfg(
    scene=scene_config,
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
    sim=simulation_config,
    viewer=viewer_config,
    decimation=1,
    episode_length_s=10.0,
)

rl_config = RslRlOnPolicyRunnerCfg(max_iterations=500, wandb_project="mjlab_cartpole")

register_mjlab_task(
    task_id="cartpole",
    env_cfg=env_config,
    play_env_cfg=env_config,
    rl_cfg=rl_config,
)
