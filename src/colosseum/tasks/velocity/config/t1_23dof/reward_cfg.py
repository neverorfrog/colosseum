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
  orientation_penalty,
  pose_deviation_penalty,
  static_stance,
  track_angular_velocity_filtered,
  track_linear_velocity_filtered,
)
from colosseum.robots.t1_23dof.constants import (
  BASE_BODY_NAME,
  FOOT_BODY_NAMES,
  FOOT_SITE_NAMES,
)

rewards = {
  # =======================
  # Task Tracking Rewards
  # =======================
  "track_linear_velocity": RewardTermCfg(
    func=track_linear_velocity_filtered,
    weight=4.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  "track_angular_velocity": RewardTermCfg(
    func=track_angular_velocity_filtered,
    weight=3.0,
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
      "swing_height": 0.09,
      "tracking_sigma": 0.005,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  # Phase 1: arms are fixed (action scale 0), so shoulder-pitch swing is
  # impossible — disable arm_phase. Re-enable when restoring arm action scale.
  # "arm_phase": RewardTermCfg(
  #   func=arm_phase,
  #   weight=1.0,
  #   params={
  #     "phase_command_name": "gait_phase",
  #     "asset_cfg": SceneEntityCfg(
  #       "robot",
  #       joint_names=("Left_Shoulder_Pitch", "Right_Shoulder_Pitch"),
  #     ),
  #     "swing_amplitude": 0.25,
  #     "max_speed": 1.5,
  #     "tracking_sigma": 0.25,
  #     "command_name": "twist",
  #   },
  # ),
  "alive": RewardTermCfg(
    func=is_alive,
    weight=0.5,
  ),
  # =========================
  # Regularization penalties
  # =========================
  "penalty_landing": RewardTermCfg(
    func=soft_landing,
    weight=-0.03,
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
  "penalty_orientation": RewardTermCfg(
    func=orientation_penalty,
    weight=-7.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  # Always-on vertical posture anchor (replaces the vertical role pose_deviation
  # played): keeps a consistent ride height without dictating joint poses.
  # Quadratic (Δh)² in meters above terrain; target = T1 spawn root z.
  "penalty_base_height": RewardTermCfg(
    func=base_height_penalty,
    weight=-20.0,
    params={"target_height": 0.66},
  ),
  "penalty_feet_ori": RewardTermCfg(
    func=foot_orientation_penalty,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "penalty_feet_yaw_diff": RewardTermCfg(
    func=feet_yaw_diff_penalty,
    weight=-3.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  # Aligns mean foot yaw to the base heading — catches the shared toe-out / yaw
  # pivot that feet_yaw_diff (feet-parallel-to-each-other) is blind to.
  "penalty_feet_yaw_mean": RewardTermCfg(
    func=feet_yaw_mean_penalty,
    weight=-3.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "penalty_action_rate": RewardTermCfg(func=action_rate_l2, weight=-1.25),
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
  "feet_slip": RewardTermCfg(
    func=feet_slip,
    weight=-5.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "sensor_name": "feet_ground_contact",
      "foot_body_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES)),
    },
  ),
  "penalty_dof_vel": RewardTermCfg(
    func=dof_vel_penalty,
    weight=-1e-3,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
  "penalty_dof_acc": RewardTermCfg(
    func=dof_acc_penalty,
    weight=-1e-6,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
}
