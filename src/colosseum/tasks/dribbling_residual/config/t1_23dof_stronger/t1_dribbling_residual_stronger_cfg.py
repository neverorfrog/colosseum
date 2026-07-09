"""Dribbling-residual "stronger" variant.

Same single-residual stack, scene, observations, and events as the base
dribbling-residual task, but tuned for a sharper kick and a behind-the-ball
approach:

  * ``twist`` circumnavigates: it aims the frozen walk at a waypoint behind the
    ball (``approach_offset``) until the robot is lined up, then commits to the
    ball, so the robot stops rushing the ball head-on and missing.
  * ``ball_vel`` speed cap raised (2.0 -> 2.5) plus a contact-triggered
    ``ball_kick_impulse`` reward for harder, target-aligned touches.
  * a dedicated inner-foot (foot5 medial capsule) ball-contact sensor + reward,
    encouraging inside-of-the-foot dribble touches.
"""

from copy import deepcopy
from dataclasses import dataclass, field

from mjlab.scene import SceneCfg

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg
from colosseum.robots.t1.sensors import FOOT5_BALL_CONTACT_SENSOR
from colosseum.tasks.dribbling_residual.mdp.ball_twist_command import (
  BallTwistCommandCfg,
)
from colosseum.tasks.dribbling_residual.mdp.ball_velocity_command import (
  BallVelocityCommandCfg,
)

from ..t1.algo_cfg import booster_t1_residual_ppo_cfg
from ..t1.cat_cfg import actions, terminations
from ..t1.curriculum_cfg import curriculum
from ..t1.event_cfg import events
from ..t1.observation_cfg import observations
from ..t1.t1_dribbling_residual_cfg import (
  booster_t1_dribbling_residual_env_cfg,
  sim_cfg,
  viewer_cfg,
)
from .reward_cfg import rewards


def scene_cfg(play: bool = False) -> SceneCfg:
  base = booster_t1_dribbling_residual_env_cfg(play).scene
  base.sensors = base.sensors + (FOOT5_BALL_CONTACT_SENSOR,)
  return base


# Reuse the base command channels, then override the two we tune.
DEFAULT_COMMANDS = dict(deepcopy(booster_t1_dribbling_residual_env_cfg().commands))
# NOTE: the twist slow zone (slow_distance/slow_speed_scale, previously 0.7/0.4
# here) was removed when BallTwistCommand was simplified.
DEFAULT_COMMANDS["twist"] = BallTwistCommandCfg(
  stop_distance=0.25,
  approach_offset=0.4,
)
DEFAULT_COMMANDS["ball_vel"] = BallVelocityCommandCfg(
  speed_range=(0.1, 2.5),
)


def booster_t1_dribbling_residual_stronger_env_cfg(
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
    cfg.curriculum.clear()

  return cfg


@register_task("t1-dribbling-residual-stronger")
@dataclass(frozen=True)
class T1DribblingResidualStrongerTask(TaskConfig):
  name: str = "t1-dribbling-residual-stronger"
  env: ColosseumEnvCfg = field(
    default_factory=booster_t1_dribbling_residual_stronger_env_cfg
  )

  @property
  def train_env_cfg(self):
    return booster_t1_dribbling_residual_stronger_env_cfg(play=False)

  @property
  def play_env_cfg(self):
    return booster_t1_dribbling_residual_stronger_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_residual_ppo_cfg()
