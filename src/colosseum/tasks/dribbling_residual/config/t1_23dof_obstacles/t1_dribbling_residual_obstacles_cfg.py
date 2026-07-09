"""Dribbling-residual task with reactive obstacle avoidance.

Same single-residual stack as the base dribbling task (frozen RMA walk + one
trainable residual), but the residual/orchestrator see an obstacle estimate and
the scene spawns a ball-path-blocking obstacle (ObstacleCommand, ball-coupled)
that refreshes when the ball_vel dribble target resamples.
"""

from copy import deepcopy
from dataclasses import dataclass, field

from mjlab.scene import SceneCfg

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
from colosseum.tasks.dribbling.mdp.obstacle_commands import (
  ObstacleCommandCfg,
)
from colosseum.tasks.dribbling.obstacle_spec import NUM_OBSTACLES, get_obstacle_cfg

from ..t1.cat_cfg import actions, terminations
from ..t1.curriculum_cfg import curriculum
from ..t1.event_cfg import events
from ..t1.t1_dribbling_residual_cfg import (
  booster_t1_dribbling_residual_env_cfg,
  sim_cfg,
  viewer_cfg,
)
from .algo_cfg import booster_t1_dribbling_residual_obstacles_ppo_cfg
from .observation_cfg import observations
from .reward_cfg import rewards


def scene_cfg(play: bool = False) -> SceneCfg:
  base = booster_t1_dribbling_residual_env_cfg(play).scene
  base.entities.update(
    {f"obstacle_{k}": get_obstacle_cfg(k) for k in range(NUM_OBSTACLES)}
  )
  return base


DEFAULT_COMMANDS = dict(deepcopy(booster_t1_dribbling_residual_env_cfg().commands))
DEFAULT_COMMANDS["adversary"] = ObstacleCommandCfg(
  num_obstacles=NUM_OBSTACLES,
  num_active=1,
  follow_command_name="ball_vel",
  behavior="static_blocker",
  forward_fraction_range=(0.35, 0.75),
  lateral_offset_range=(-0.3, 0.3),
)


def booster_t1_dribbling_residual_obstacles_env_cfg(
  play: bool = False,
) -> ColosseumEnvCfg:
  cfg = ColosseumEnvCfg(
    scene=scene_cfg(play),
    observations=observations,
    actions=actions,
    commands=DEFAULT_COMMANDS,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics={},
    viewer=viewer_cfg(),
    sim=sim_cfg(),
    decimation=4,
    episode_length_s=30.0,
    encoders={},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["loco_actor"].enable_corruption = False
    cfg.observations["dribble_actor"].enable_corruption = False
    cfg.observations["obstacle_residual"].enable_corruption = False
    cfg.curriculum.clear()
    # cfg.events.clear()

  return cfg


@register_task("t1-dribbling-residual-obstacles")
@dataclass(frozen=True)
class T1DribblingResidualObstaclesTask(TaskConfig):
  name: str = "t1-dribbling-residual-obstacles"
  env: ColosseumEnvCfg = field(
    default_factory=booster_t1_dribbling_residual_obstacles_env_cfg
  )

  @property
  def train_env_cfg(self):
    return self.env

  @property
  def play_env_cfg(self):
    return booster_t1_dribbling_residual_obstacles_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_dribbling_residual_obstacles_ppo_cfg()
