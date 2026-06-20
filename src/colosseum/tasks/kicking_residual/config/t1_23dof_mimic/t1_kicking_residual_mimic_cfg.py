"""Residual kicking task + motion-mimic: t1-kicking-residual-mimic.

Same residual-PPO setup as ``t1-kicking-residual`` (frozen walk branch reading
``loco_actor``, steered toward the ball via ``BallTwistCommand``, plus a
trainable residual ``kick_actor`` branch + orchestrator), with an added
``"motion"`` command (``GatedHoldMotionCommand``): the reference kick clip
(``models/trajectories/t1_motion.npz``) stays on frame 0 until the robot is
within 0.25 m of the ball (the same threshold at which the twist command
zeroes out), then plays once and holds its final frame. 6 gated
motion-tracking rewards give the residual an imitation signal for *how* to
kick, active only once near the ball.

``scene_cfg``/``viewer_cfg``/``sim_cfg``/``algo_cfg``/``curriculum``/``events``
are unchanged, imported from the ``t1_23dof`` sibling.
"""

import math
from dataclasses import dataclass, field

from mjlab.envs.mdp.events import (
  reset_root_state_uniform,
)
from mjlab.managers import (
  EventTermCfg,
  SceneEntityCfg,
)

from colosseum.config.types.task import TaskConfig, register_task
from colosseum.envs.colosseum_env import ColosseumEnvCfg

from ..t1_23dof.algo_cfg import booster_t1_residual_ppo_cfg
from ..t1_23dof.event_cfg import events
from ..t1_23dof.t1_kicking_residual_cfg import scene_cfg, sim_cfg, viewer_cfg
from .cat_cfg import actions, commands, terminations
from .observation_cfg import observations
from .reward_cfg import rewards


def booster_t1_kicking_residual_mimic_env_cfg(play: bool = False) -> ColosseumEnvCfg:
  cfg = ColosseumEnvCfg(
    scene=scene_cfg(play),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics={},
    viewer=viewer_cfg(),
    sim=sim_cfg(),
    decimation=4,
    episode_length_s=30.0,
    encoders={},
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = False

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["loco_actor"].enable_corruption = False
    cfg.observations["kick_actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events = {}
    cfg.events["reset_base"] = EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.0, 0.0),
          "y": (-0.0, 0.0),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    )

    cfg.events["reset_ball"] = EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("ball"),
        "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (0.11, 0.11)},
        "velocity_range": {},
      },
    )

    gait = cfg.commands["gait_phase"]
    gait.randomize_phase = False

    ball_angle = cfg.commands["ball_angle"]
    ball_angle.angle_offset_range = (-math.pi/4, math.pi/4)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


@register_task("t1-kicking-residual-mimic")
@dataclass(frozen=True)
class T1KickingResidualMimicTask(TaskConfig):
  name: str = "t1-kicking-residual-mimic"
  env: ColosseumEnvCfg = field(default_factory=booster_t1_kicking_residual_mimic_env_cfg)

  @property
  def train_env_cfg(self):
    return booster_t1_kicking_residual_mimic_env_cfg(play=False)

  @property
  def play_env_cfg(self):
    return booster_t1_kicking_residual_mimic_env_cfg(play=True)

  @property
  def algo_cfg(self):
    return booster_t1_residual_ppo_cfg()
