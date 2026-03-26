import math

from mjlab.envs.mdp import (
  action_rate_l2,
  joint_pos_limits,
)
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  angular_momentum_penalty,
  body_angular_velocity_penalty,
  feet_clearance,
  feet_slip,
  feet_swing_height,
  flat_orientation,
  self_collision_cost,
  soft_landing,
)

from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_SITE_NAMES
from colosseum.robots.t1_23dof.sensors import SELF_COLLISION_SENSOR
from colosseum.tasks.dribbling.mdp.rewards import (
  ball_vel_angle,
  ball_vel_norm,
  ball_vel_tracking,
  robot_ball_approach_vel,
  robot_ball_distance,
  robot_ball_yaw,
)

rewards = {
  # ------------------------------------------------------------------ #
  # Task rewards                                                         #
  # ------------------------------------------------------------------ #
  "ball_vel_tracking": RewardTermCfg(
    func=ball_vel_tracking,
    weight=0.5,
    params={"command_name": "ball_vel", "sharpness": 1.0},
  ),
  "ball_vel_norm": RewardTermCfg(
    func=ball_vel_norm,
    weight=4.0,
    params={"command_name": "ball_vel", "sharpness": 1.0},
  ),
  "ball_vel_angle": RewardTermCfg(
    func=ball_vel_angle,
    weight=4.0,
    params={"command_name": "ball_vel"},
  ),
  "robot_ball_distance": RewardTermCfg(
    func=robot_ball_distance,
    weight=4.0,
    params={},
  ),
  "robot_ball_yaw": RewardTermCfg(
    func=robot_ball_yaw,
    weight=4.0,
    params={"command_name": "ball_vel"},
  ),
  "robot_ball_approach_vel": RewardTermCfg(
    func=robot_ball_approach_vel,
    weight=0.5,
    params={"command_name": "ball_vel"},
  ),
  # ------------------------------------------------------------------ #
  # Locomotion regularization                                            #
  # ------------------------------------------------------------------ #
  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=1.0,
    params={
      "std": math.sqrt(0.2),
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME)),
    },
  ),
  "body_ang_vel": RewardTermCfg(
    func=body_angular_velocity_penalty,
    weight=-0.05,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "angular_momentum": RewardTermCfg(
    func=angular_momentum_penalty,
    weight=-0.02,
    params={"sensor_name": "robot/root_angmom"},
  ),
  "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
  "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.1),
  "foot_clearance": RewardTermCfg(
    func=feet_clearance,
    weight=-2.0,
    params={
      "target_height": 0.1,
      "command_name": "ball_vel",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
    },
  ),
  "foot_swing_height": RewardTermCfg(
    func=feet_swing_height,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "target_height": 0.1,
      "command_name": "ball_vel",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
    },
  ),
  "foot_slip": RewardTermCfg(
    func=feet_slip,
    weight=-0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "ball_vel",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
    },
  ),
  "soft_landing": RewardTermCfg(
    func=soft_landing,
    weight=-1e-5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "ball_vel",
      "command_threshold": 0.05,
    },
  ),
  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-1.0,
    params={"sensor_name": SELF_COLLISION_SENSOR.name, "force_threshold": 10.0},
  ),
}
