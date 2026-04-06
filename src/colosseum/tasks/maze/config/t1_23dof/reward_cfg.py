"""Reward configuration for T1 maze task."""

import math

from mjlab.envs.mdp import action_rate_l2, joint_pos_limits
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  angular_momentum_penalty,
  body_angular_velocity_penalty,
  feet_clearance,
  feet_slip,
  feet_swing_height,
  self_collision_cost,
  soft_landing,
)

from colosseum.mdp.rewards import flat_orientation
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_SITE_NAMES
from colosseum.robots.t1_23dof.sensors import SELF_COLLISION_SENSOR
from colosseum.tasks.maze.mdp.rewards import (
  track_angular_velocity,
  track_velocity_direction,
  wall_collisions,
)

rewards: dict[str, RewardTermCfg] = {
  # ------------------------------------------------------------------ #
  # Survival                                                             #
  # ------------------------------------------------------------------ #
  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=0.7,
    params={
      "std": math.sqrt(0.5),
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
    },
  ),
  # ------------------------------------------------------------------ #
  # Task rewards                                                         #
  # ------------------------------------------------------------------ #
  "track_velocity_direction": RewardTermCfg(
    func=track_velocity_direction,
    weight=2.0,
    params={
      "command_name": "velocity",
      "use_body_frame": True,
    },
  ),
  "track_angular_velocity": RewardTermCfg(
    func=track_angular_velocity,
    weight=1.0,
    params={"command_name": "velocity"},
  ),
  # ------------------------------------------------------------------ #
  # Regularization                                                       #
  # ------------------------------------------------------------------ #
  "body_ang_vel": RewardTermCfg(
    func=body_angular_velocity_penalty,
    weight=-0.05,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,))},
  ),
  "angular_momentum": RewardTermCfg(
    func=angular_momentum_penalty,
    weight=-0.5,
    params={"sensor_name": "robot/root_angmom"},
  ),
  "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
  "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.05),
  "wall_collisions": RewardTermCfg(
    func=wall_collisions,
    weight=-10.0,
    params={"sensor_name": "wall_collision"},
  ),
  "foot_clearance": RewardTermCfg(
    func=feet_clearance,
    weight=-2.0,
    params={
      "target_height": 0.1,
      "height_sensor_name": "foot_height_scan",
      "command_name": "velocity",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
    },
  ),
  "foot_swing_height": RewardTermCfg(
    func=feet_swing_height,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.1,
      "command_name": "velocity",
      "command_threshold": 0.05,
    },
  ),
  "foot_slip": RewardTermCfg(
    func=feet_slip,
    weight=-0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "velocity",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
    },
  ),
  "soft_landing": RewardTermCfg(
    func=soft_landing,
    weight=-1e-5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "velocity",
      "command_threshold": 0.05,
    },
  ),
  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-1.0,
    params={"sensor_name": SELF_COLLISION_SENSOR.name, "force_threshold": 10.0},
  ),
}
