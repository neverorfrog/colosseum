import math  # noqa: F401

from mjlab.envs.mdp import (
  action_rate_l2,
  joint_pos_limits,
)
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
from colosseum.robots.t1_23dof.constants import (
  BASE_BODY_NAME,
  FOOT_SITE_NAMES,
)
from colosseum.robots.t1_23dof.sensors import (
  FOOT_FOOT_CONTACT_SENSOR,
  NONFOOT_BALL_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
)
from colosseum.tasks.dribbling.mdp.rewards import (
  ball_protection_gated,
  ball_vel_angle_body,
  ball_vel_norm,
  ball_vel_tracking_body,
  feet_distance_penalty,
  obstacle_avoidance,
  pose_deviation,
  robot_ball_approach_vel_gated,
  robot_ball_distance_gated,
  robot_ball_yaw_body,
  stance_phase_schedule,
  swing_phase_schedule,
)

rewards = {
  # ------------------------------------------------------------------ #
  # Task rewards                                                         #
  # ------------------------------------------------------------------ #
  "ball_vel_tracking": RewardTermCfg(
    func=ball_vel_tracking_body,
    weight=2.0,
    params={"command_name": "ball_vel", "sharpness": 1.0},
  ),
  "ball_vel_norm": RewardTermCfg(
    func=ball_vel_norm,
    weight=4.0,
    params={"command_name": "ball_vel", "sharpness": 1.0},
  ),
  "ball_vel_angle": RewardTermCfg(
    func=ball_vel_angle_body,
    weight=4.0,
    params={"command_name": "ball_vel"},
  ),
  "robot_ball_distance": RewardTermCfg(
    func=robot_ball_distance_gated,
    weight=0.3,
    params={
      "command_name": "adversary",
      "ball_vel_command_name": "ball_vel",
      "sharpness_base": 0.5,
      "sharpness_tight": 2.0,
      "ball_far_loosening": 0.7,
      "ball_far_threshold": 1.0,
    },
  ),
  "robot_ball_yaw": RewardTermCfg(
    func=robot_ball_yaw_body,
    weight=4.0,
    params={"command_name": "ball_vel"},
  ),
  "robot_ball_approach_vel": RewardTermCfg(
    func=robot_ball_approach_vel_gated,
    weight=0.5,
    params={
      "command_name": "adversary",
      "ball_vel_command_name": "ball_vel",
      "ball_far_threshold": 1.0,
    },
  ),
  "obstacle_avoidance": RewardTermCfg(
    func=obstacle_avoidance,
    weight=-1.0,
    params={
      "command_name": "adversary",
      "ball_vel_command_name": "ball_vel",
      "detection_range": 3.0,
      "collision_near_distance": 0.5,
      "collision_far_distance": 1.5,
      "direction_sharpness": 4.0,
      "collision_weight": 0.2,
      "direction_weight": 1.0,
      "min_cmd_speed": 0.05,
      "cmd_speed_ref": 1.0,
      "ball_engagement_radius": 1.0,
    },
  ),
  # "ball_protection": RewardTermCfg(
  #   func=ball_protection_gated,
  #   weight=1.0,
  #   params={
  #     "command_name": "adversary",
  #     "ball_vel_command_name": "ball_vel",
  #     "activation_radius": 3.0,
  #   },
  # ),
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
    weight=-0.5,
    params={"sensor_name": "robot/root_angmom"},
  ),
  "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
  "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.1),
  "foot_swing_height": RewardTermCfg(
    func=feet_swing_height,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.1,
      "command_name": "ball_vel",
      "command_threshold": 0.05,
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
  "feet_distance": RewardTermCfg(
    func=feet_distance_penalty,
    weight=-4.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_dist": 0.15,
    },
  ),
  "foot_foot_contact": RewardTermCfg(
    func=self_collision_cost,
    weight=-3.0,
    params={"sensor_name": FOOT_FOOT_CONTACT_SENSOR.name, "force_threshold": 1.0},
  ),
  "nonfoot_ball_contact": RewardTermCfg(
    func=self_collision_cost,
    weight=-2.0,
    params={"sensor_name": NONFOOT_BALL_CONTACT_SENSOR.name, "force_threshold": 1.0},
  ),
  # ------------------------------------------------------------------ #
  # Phase-schedule feet rewards                                          #
  # ------------------------------------------------------------------ #
  "swing_phase": RewardTermCfg(
    func=swing_phase_schedule,
    weight=4.0,
    params={
      "phase_command_name": "gait_phase",
      "sensor_name": "feet_ground_contact",
    },
  ),
  "stance_phase": RewardTermCfg(
    func=stance_phase_schedule,
    weight=4.0,
    params={
      "phase_command_name": "gait_phase",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  ),
  # ------------------------------------------------------------------ #
  # Pose deviation                                                       #
  # ------------------------------------------------------------------ #
  "pose_arms": RewardTermCfg(
    func=pose_deviation,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          r"(?i).*shoulder.*",
          r"(?i).*elbow.*",
        ),
      ),
      "std": 0.1,
    },
  ),
  "pose_legs": RewardTermCfg(
    func=pose_deviation,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          r"(?i).*hip.*",
          r"(?i).*knee.*",
          r"(?i).*ankle.*",
        ),
      ),
      "std": 0.3,
    },
  ),
}
