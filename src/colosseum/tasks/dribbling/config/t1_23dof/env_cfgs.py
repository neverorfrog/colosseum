"""Booster T1 dribbling environment configurations.

The dribbling task builds on the flat velocity task but replaces random
velocity commands with a BallApproachCommand that points the robot toward
a soccer ball.  The ball is a free-floating entity added to the scene and
is respawned at a random position (1–3 m from the robot) on every episode.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp import events as envs_mdp_events
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from colosseum.assets.ball.ball_spec import BALL_RADIUS, get_ball_cfg
from colosseum.robots.t1_23dof.constants import (
  ACTION_SCALE,
  FOOT_GEOM_NAMES,
  get_robot_cfg,
)
from colosseum.robots.t1_23dof.contacts import (
  FEET_GROUND_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
)
from colosseum.tasks.dribbling.mdp import BallApproachCommandCfg
from colosseum.tasks.velocity.config.t1_23dof.env_cfgs import booster_t1_flat_env_cfg


def booster_t1_dribbling_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 dribbling task configuration.

  Starts from the flat velocity config and:
  - Adds a soccer ball as a free-floating scene entity.
  - Replaces the random velocity command with BallApproachCommand.
  - Adds a reset event that respawns the ball at a random position each episode.
  """
  cfg = booster_t1_flat_env_cfg(play=play)

  # ------------------------------------------------------------------ #
  # Scene: add ball entity                                               #
  # ------------------------------------------------------------------ #
  cfg.scene.entities["ball"] = get_ball_cfg()

  # ------------------------------------------------------------------ #
  # Command: replace random velocity with ball-approach command          #
  # ------------------------------------------------------------------ #
  assert cfg.commands is not None
  cfg.commands["twist"] = BallApproachCommandCfg(
    robot_entity="robot",
    ball_entity="ball",
    base_velocity=0.8,
    min_velocity=0.1,
    max_velocity=1.5,
    ema_smoothing=0.2,
    body_forward_axis=(1.0, 0.0, 0.0),
    angular_velocity_gain=2.0,
    max_angular_velocity=1.5,
    min_alignment_scale=0.1,
    debug_vis=True,
  )

  # ------------------------------------------------------------------ #
  # Events: respawn ball at random position each episode reset           #
  # ------------------------------------------------------------------ #
  cfg.events["reset_ball"] = EventTermCfg(
    func=envs_mdp_events.reset_root_state_uniform,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("ball"),
      # Random XY offset from env origin; Z fixed at ball radius (on the ground).
      "pose_range": {
        "x": (-3.0, 3.0),
        "y": (-3.0, 3.0),
      },
      "velocity_range": {},
    },
  )

  # Disable curriculum for dribbling (commands are determined by ball position,
  # not a velocity curriculum).
  if cfg.curriculum is not None:
    cfg.curriculum.pop("command_vel", None)

  if play:
    cfg.episode_length_s = int(1e9)

  return cfg
