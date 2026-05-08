import math

from mjlab.envs.mdp import (
  action_rate_l2,
  is_alive,
  joint_pos_limits,
)
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  angular_momentum_penalty,
  body_angular_velocity_penalty,
  feet_clearance,
  feet_swing_height,
  self_collision_cost,
  track_angular_velocity,
  track_linear_velocity,
  variable_posture,
)

from colosseum.mdp.rewards import (
  base_height_penalty,
  feet_distance_penalty,
  feet_phase,
  flat_orientation,
  foot_orientation_penalty,
  orientation_penalty,
  pose_deviation_penalty,
  stance_phase_schedule,
  static_stance,
  swing_phase_schedule,
)
from colosseum.robots.t1_23dof.constants import (
  BASE_BODY_NAME,
  FOOT_BODY_NAMES,
  FOOT_SITE_NAMES,
)
from colosseum.robots.t1_23dof.sensors import SELF_COLLISION_SENSOR

rewards = {
  # =======================
  # Task Tracking Rewards
  # =======================
  "track_linear_velocity": RewardTermCfg(
    func=track_linear_velocity,
    weight=3.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  "track_angular_velocity": RewardTermCfg(
    func=track_angular_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  # =========================
  # Survival rewards
  # =========================
  "alive": RewardTermCfg(
    func=is_alive,
    weight=0.5,
  ),
  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=0.5,
    params={
      "std": math.sqrt(0.5),
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME)),
    },
  ),
  # =========================
  # Regularization penalties
  # =========================
  "penalty_base_height": RewardTermCfg(
    func=base_height_penalty,
    weight=-20.0,
    params={
      "target_height": 0.6,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  ),
  "penalty_body_ang_vel": RewardTermCfg(
    func=body_angular_velocity_penalty,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "penalty_orientation": RewardTermCfg(
    func=orientation_penalty,
    weight=-10.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "penalty_feet_ori": RewardTermCfg(
    func=foot_orientation_penalty,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "penalty_action_rate": RewardTermCfg(func=action_rate_l2, weight=-2.0),
  "pose_deviation": RewardTermCfg(
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
  # "penalty_pose_deviation": RewardTermCfg(
  #   func=pose_deviation_penalty,
  #   weight=-0.5,
  #   params={
  #     "asset_cfg": SceneEntityCfg(
  #       "robot",
  #       joint_names=(
  #         "Waist",
  #         r"(?i).*shoulder.*",
  #         r"(?i).*elbow.*",
  #         r"(?i).*head.*",
  #       ),
  #     ),
  #     "pose_weights": [
  #       # Joints resolved in JOINT_NAMES order: Waist(12), L/R Shoulder(13-14,17-18),
  #       # L/R Elbow(15-16,19-20), AAHead_yaw(21), Head_pitch(22).
  #       3.0,  # Waist
  #       5.0,  # Left_Shoulder_Pitch
  #       5.0,  # Left_Shoulder_Roll
  #       3.0,  # Left_Elbow_Pitch
  #       3.0,  # Left_Elbow_Yaw
  #       5.0,  # Right_Shoulder_Pitch
  #       5.0,  # Right_Shoulder_Roll
  #       3.0,  # Right_Elbow_Pitch
  #       3.0,  # Right_Elbow_Yaw
  #       0.5,  # AAHead_yaw
  #       0.5,  # Head_pitch
  #     ],
  #   },
  # ),
  "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
  "penalty_feet_distance": RewardTermCfg(
    func=feet_distance_penalty,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "min_dist": 0.15,
    },
  ),
  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-5.0,
    params={"sensor_name": SELF_COLLISION_SENSOR.name, "force_threshold": 10.0},
  ),
  # =========================
  # Foot trajectory Rewards
  # =========================
  # "feet_phase": RewardTermCfg(
  #   func=feet_phase,
  #   weight=5.0,
  #   params={
  #     "phase_command_name": "gait_phase",
  #     "height_sensor_name": "foot_height_scan",
  #     "swing_height": 0.09,
  #     "tracking_sigma": 0.008,
  #   },
  # ),
  "swing_phase": RewardTermCfg(
    func=swing_phase_schedule,
    weight=3.0,
    params={
      "phase_command_name": "gait_phase",
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "stance_phase": RewardTermCfg(
    func=stance_phase_schedule,
    weight=2.0,
    params={
      "phase_command_name": "gait_phase",
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "static_stance": RewardTermCfg(
    func=static_stance,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "foot_clearance": RewardTermCfg(
    func=feet_clearance,
    weight=-2.0,
    params={
      "target_height": 0.1,
      "height_sensor_name": "foot_height_scan",
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
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.1,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
}


# Rationale for std values (pose_deviation / variable_posture, whole body):
# - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
# - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
# - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
# - Waist roll/pitch stay tight to keep the torso upright and stable.
# - Shoulders/elbows get moderate freedom for natural arm swing during walking.
# Running values are ~1.5-2x walking values to accommodate larger motion range.
# Joint naming convention in T1_23dof: Left_Hip_Pitch, Right_Knee_Pitch, Waist, etc.
# Patterns use (?i) for case-insensitive matching.
rewards["pose_deviation"].params["std_standing"] = {".*": 0.05}
rewards["pose_deviation"].params["std_walking"] = {
  # Lower body.
  r"(?i).*hip_pitch.*": 0.3,
  r"(?i).*hip_roll.*": 0.15,
  r"(?i).*hip_yaw.*": 0.15,
  r"(?i).*knee.*": 0.35,
  r"(?i).*ankle_pitch.*": 0.25,
  r"(?i).*ankle_roll.*": 0.1,
  # Waist.
  r"Waist": 0.1,
  # Arms.
  r"(?i).*shoulder_pitch.*": 0.15,
  r"(?i).*shoulder_roll.*": 0.15,
  r"(?i).*elbow.*": 0.15,
  # Neck.
  r"(?i).*head.*": 0.2,
}
rewards["pose_deviation"].params["std_running"] = {
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
