import math

from mjlab.envs.mdp import (
  action_rate_l2,
  is_alive,
  joint_pos_limits,
)
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  body_angular_velocity_penalty,
  soft_landing,
)

from colosseum.mdp.rewards import (
  arm_phase,
  base_height_penalty,
  dof_acc_penalty,
  dof_vel_penalty,
  feet_distance_penalty,
  feet_phase,
  feet_slip,
  feet_swing,
  feet_yaw_diff_penalty,
  feet_yaw_mean_penalty,
  foot_orientation_penalty,
  lin_vel_z_filtered_penalty,
  orientation_penalty,
  pose_deviation_penalty,
  power_penalty,
  static_stance,
  torque_tiredness_penalty,
  torques_penalty,
  track_ang_vel_yaw_filtered,
  track_lin_vel_axis_filtered,
)
from colosseum.robots.t1_23dof.constants import (
  BASE_BODY_NAME,
  FOOT_BODY_NAMES,
  FOOT_SITE_NAMES,
)

LEG_JOINT_PATTERNS = (".*Hip.*", ".*Knee.*", ".*Ankle.*")
LOWER_BODY_JOINT_PATTERNS = (".*Hip.*", ".*Knee.*", ".*Ankle.*", "Waist")
ALL_JOINTS_PATTERNS = ("^(?!AAHead_yaw$|Head_pitch$).*$",)

# Leg torque limits (Nm), matching LOCOMOTION_ACTUATORS (booster_gym URDF efforts).
LEG_EFFORT_LIMITS = {
  f"{side}_{joint}": limit
  for side in ("Left", "Right")
  for joint, limit in (
    ("Hip_Pitch", 45.0),
    ("Hip_Roll", 30.0),
    ("Hip_Yaw", 30.0),
    ("Knee_Pitch", 60.0),
    ("Ankle_Pitch", 24.0),
    ("Ankle_Roll", 15.0),
  )
}

rewards = {
  # =======================
  # Task Tracking Rewards
  # =======================
  # Per-axis kernels (booster_gym form): each axis earns reward and gradient
  # independently. The combined-product form (track_*_velocity_filtered) zeroes
  # the gradient on every axis whenever one axis is far off.
  "track_lin_vel_x": RewardTermCfg(
    func=track_lin_vel_axis_filtered,
    weight=4.0,
    params={"command_name": "twist", "std": math.sqrt(0.25), "axis": 0},
  ),
  "track_lin_vel_y": RewardTermCfg(
    func=track_lin_vel_axis_filtered,
    weight=4.0,
    params={"command_name": "twist", "std": math.sqrt(0.25), "axis": 1},
  ),
  "track_ang_vel_yaw": RewardTermCfg(
    func=track_ang_vel_yaw_filtered,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  "feet_swing": RewardTermCfg(
    func=feet_swing,
    weight=3.0,
    params={
      "phase_command_name": "gait_phase",
      "sensor_name": "feet_ground_contact",
      "swing_period": 0.2,
      "contact_threshold": 0.1,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "feet_phase": RewardTermCfg(
    func=feet_phase,
    weight=2.0,
    params={
      "phase_command_name": "gait_phase",
      "height_sensor_name": "foot_height_scan",
      "swing_height": 0.11,
      "tracking_sigma": 0.005,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "arm_phase": RewardTermCfg(
    func=arm_phase,
    weight=2.0,
    params={
      "phase_command_name": "gait_phase",
      # (left, right) pairs; shoulder swings wide, elbow follows with a smaller amplitude.
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          "Left_Shoulder_Pitch",
          "Right_Shoulder_Pitch",
          "Left_Elbow_Pitch",
          "Right_Elbow_Pitch",
        ),
        # Keep (left, right) pair order; otherwise ids resolve to global-index
        # order [L_Sh, L_El, R_Sh, R_El], breaking contralateral pairing.
        preserve_order=True,
      ),
      "swing_amplitude": (0.25, 0.25, 0.15, 0.15),
      "max_speed": 1.5,
      "tracking_sigma": 0.25,
      "command_name": "twist",
    },
  ),
  "alive": RewardTermCfg(
    func=is_alive,
    weight=0.25,
  ),
  # =========================
  # Regularization penalties
  # =========================
  "penalty_landing": RewardTermCfg(
    func=soft_landing,
    weight=-0.01,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "penalty_body_ang_vel": RewardTermCfg(
    func=body_angular_velocity_penalty,
    weight=-2.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "penalty_lin_vel_z": RewardTermCfg(
    func=lin_vel_z_filtered_penalty,
    weight=-0.1,
    params={"command_name": "twist"},
  ),
  "penalty_orientation": RewardTermCfg(
    func=orientation_penalty,
    weight=-20.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  # Always-on vertical posture anchor (replaces the vertical role pose_deviation
  # played): keeps a consistent ride height without dictating joint poses.
  # Quadratic (Δh)² in meters above terrain; target = T1 spawn root z.
  "penalty_base_height": RewardTermCfg(
    func=base_height_penalty,
    weight=-15.0,
    params={"target_height": 0.64},
  ),
  "penalty_feet_ori": RewardTermCfg(
    func=foot_orientation_penalty,
    weight=-2.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "penalty_feet_yaw_diff": RewardTermCfg(
    func=feet_yaw_diff_penalty,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "penalty_feet_yaw_mean": RewardTermCfg(
    func=feet_yaw_mean_penalty,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "penalty_pose_deviation": RewardTermCfg(
    func=pose_deviation_penalty,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_PATTERNS),
      "weights_standing": {
        ".*Shoulder_Pitch": 1.0,
        ".*Shoulder_Roll": 20.0,
        ".*Elbow_Pitch": 1.0,
        ".*Elbow_Yaw": 20.0,
        "Waist": 20.0,
        ".*Hip_Pitch": 0.01,
        ".*Hip_Roll": 1.0,
        ".*Hip_Yaw": 5.0,
        ".*Knee_Pitch": 0.01,
        ".*Ankle_Pitch": 5.0,
        ".*Ankle_Roll": 5.0,
      },
    },
  ),
  "penalty_action_rate": RewardTermCfg(func=action_rate_l2, weight=-1.0),
  "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
  "penalty_feet_distance": RewardTermCfg(
    func=feet_distance_penalty,
    weight=-5.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "min_dist": 0.15,
    },
  ),
  # site_names → linear slip; body_names → foot yaw-rate for the rotational scrub.
  "penalty_feet_slip": RewardTermCfg(
    func=feet_slip,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "sensor_name": "feet_ground_contact",
      "foot_body_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES)),
    },
  ),
  "penalty_dof_vel": RewardTermCfg(
    func=dof_vel_penalty,
    weight=-1e-3,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_PATTERNS)},
  ),
  "penalty_dof_acc": RewardTermCfg(
    func=dof_acc_penalty,
    weight=-1e-6,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_PATTERNS)},
  ),
  # Effort penalties (booster_gym weights): price torque magnitude, proximity to
  # the torque limit, and positive mechanical power. Legs only (upper body is
  # PD-held and its holding torque is not under policy control).
  "penalty_torques": RewardTermCfg(
    func=torques_penalty,
    weight=-2e-5,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_PATTERNS)},
  ),
  "penalty_torque_tiredness": RewardTermCfg(
    func=torque_tiredness_penalty,
    weight=-1e-3,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS),
      "effort_limits": LEG_EFFORT_LIMITS,
    },
  ),
  "penalty_power": RewardTermCfg(
    func=power_penalty,
    weight=-2e-5,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_PATTERNS)},
  ),
}
