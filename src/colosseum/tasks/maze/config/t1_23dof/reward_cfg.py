"""Reward configuration for T1 maze task."""

import math

from mjlab.envs.mdp import action_rate_l2, joint_pos_limits
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  angular_momentum_penalty,
  body_angular_velocity_penalty,
  feet_swing_height,
  self_collision_cost,
  soft_landing,
)

from colosseum.mdp.rewards import flat_orientation
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_SITE_NAMES
from colosseum.robots.t1_23dof.sensors import SELF_COLLISION_SENSOR
from colosseum.tasks.maze.mdp.rewards import (
  contact_force_penalty,
  distance_to_next_cell_shaping,
  goal_reward,
  track_angular_velocity,
  track_velocity_direction,
  wall_collisions,
  z_velocity,
)

rewards: dict[str, RewardTermCfg] = {
  # ------------------------------------------------------------------ #
  # Survival                                                             #
  # ------------------------------------------------------------------ #
  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=1.0,
    params={
      "std": math.sqrt(0.2),
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
    },
  ),
  # ------------------------------------------------------------------ #
  # Task rewards                                                         #
  # ------------------------------------------------------------------ #
  "goal": RewardTermCfg(
    func=goal_reward,
    weight=10.0,
    params={"threshold": 1.5},
  ),
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
    weight=0.5,
    params={"command_name": "velocity"},
  ),
  "distance_shaping": RewardTermCfg(
    func=distance_to_next_cell_shaping,
    weight=0.5,
    params={"abstraction_name": "grid"},
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
    weight=-2.0,
    params={"sensor_name": "wall_collision"},
  ),
  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-1.0,
    params={"sensor_name": SELF_COLLISION_SENSOR.name, "force_threshold": 10.0},
  ),
  "z_velocity": RewardTermCfg(
    func=z_velocity,
    weight=-0.5,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=("root_site",))},
  ),
}
