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
  feet_air_time,
  feet_clearance,
  feet_slip,
  feet_swing_height,
  flat_orientation,
  self_collision_cost,
  soft_landing,
  track_angular_velocity,
  track_linear_velocity,
  variable_posture,
)

from colosseum.robots.t1_23dof.sensors import SELF_COLLISION_SENSOR
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_SITE_NAMES

rewards = {
  "track_linear_velocity": RewardTermCfg(
    func=track_linear_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  "track_angular_velocity": RewardTermCfg(
    func=track_angular_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.5)},
  ),
  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=1.0,
    params={
      "std": math.sqrt(0.2),
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME)),
    },
  ),
  "pose": RewardTermCfg(
    func=variable_posture,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "command_name": "twist",
      "std_standing": {},  # Set per-robot.
      "std_walking": {},  # Set per-robot.
      "std_running": {},  # Set per-robot.
      "walking_threshold": 0.05,
      "running_threshold": 1.5,
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
  "air_time": RewardTermCfg(
    func=feet_air_time,
    weight=0.0,  # Override per-robot.
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.05,
      "threshold_max": 0.5,
      "command_name": "twist",
      "command_threshold": 0.5,
    },
  ),
  "foot_clearance": RewardTermCfg(
    func=feet_clearance,
    weight=-2.0,
    params={
      "target_height": 0.1,
      "command_name": "twist",
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
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
    },
  ),
  "foot_slip": RewardTermCfg(
    func=feet_slip,
    weight=-0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
    },
  ),
  "soft_landing": RewardTermCfg(
    func=soft_landing,
    weight=-1e-5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-1.0,
    params={"sensor_name": SELF_COLLISION_SENSOR.name, "force_threshold": 10.0},
  ),
}


# Rationale for std values:
# - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
# - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
# - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
# - Waist roll/pitch stay tight to keep the torso upright and stable.
# - Shoulders/elbows get moderate freedom for natural arm swing during walking.
# - Wrists are loose (0.3) since they don't affect balance much.
# Running values are ~1.5-2x walking values to accommodate larger motion range.
# Joint naming convention in T1_23dof: Left_Hip_Pitch, Right_Knee_Pitch, Waist, etc.
# Patterns use (?i) for case-insensitive matching.
# Running values are ~1.5-2x walking to accommodate larger motion range.
rewards["pose"].params["std_standing"] = {".*": 0.05}
rewards["pose"].params["std_walking"] = {
  # Lower body.
  r"(?i).*hip_pitch.*": 0.3,
  r"(?i).*hip_roll.*": 0.15,
  r"(?i).*hip_yaw.*": 0.15,
  r"(?i).*knee.*": 0.35,
  r"(?i).*ankle_pitch.*": 0.25,
  r"(?i).*ankle_roll.*": 0.1,
  # Waist (single joint).
  r"Waist": 0.1,
  # Arms.
  r"(?i).*shoulder_pitch.*": 0.15,
  r"(?i).*shoulder_roll.*": 0.15,
  r"(?i).*elbow.*": 0.15,
  # Neck.
  r"(?i).*head.*": 0.2,
}
rewards["pose"].params["std_running"] = {
  # Lower body.
  r"(?i).*hip_pitch.*": 0.5,
  r"(?i).*hip_roll.*": 0.2,
  r"(?i).*hip_yaw.*": 0.2,
  r"(?i).*knee.*": 0.6,
  r"(?i).*ankle_pitch.*": 0.35,
  r"(?i).*ankle_roll.*": 0.15,
  # Waist.
  r"Waist": 0.2,
  # Arms.
  r"(?i).*shoulder_pitch.*": 0.5,
  r"(?i).*shoulder_roll.*": 0.2,
  r"(?i).*elbow.*": 0.35,
  # Neck.
  r"(?i).*head.*": 0.3,
}
