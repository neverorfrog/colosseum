"""Reward config for the booster-mimic Waist+legs task.

Identical to the 12-DOF task's reward set (booster_gym's T1.yaml scales), except
the Waist joint is added to the torque/power/effort regularization since it is
now actuated. Orientation and base angular-velocity weights are kept identical
to the 12-DOF task so this run is a clean A/B on the effect of the waist DOF.
"""

import math

from mjlab.envs.mdp import action_rate_l2, is_alive, joint_pos_limits
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import body_angular_velocity_penalty

from colosseum.mdp.rewards import (
  base_height_penalty,
  dof_acc_penalty,
  dof_vel_penalty,
  feet_distance_penalty,
  feet_slip,
  feet_swing,
  feet_yaw_diff_penalty,
  feet_yaw_mean_penalty,
  foot_orientation_penalty,
  lin_vel_z_filtered_penalty,
  orientation_penalty,
  power_penalty,
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

# Actuated joints (Waist + legs): regularize torque/power/effort over all of them.
ACTUATED_JOINT_PATTERNS = ("Waist", ".*Hip.*", ".*Knee.*", ".*Ankle.*")

# Actuated torque limits (Nm): legs match booster T1_locomotion; Waist = 30.
ACTUATED_EFFORT_LIMITS = {
  "Waist": 30.0,
  **{
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
  },
}

_TRACKING_SIGMA = math.sqrt(0.25)  # booster tracking_sigma = 0.25 (=std^2)

rewards = {
  # ----------------------- task tracking ----------------------- #
  "survival": RewardTermCfg(func=is_alive, weight=0.25),
  "tracking_lin_vel_x": RewardTermCfg(
    func=track_lin_vel_axis_filtered,
    weight=1.0,
    params={"command_name": "twist", "std": _TRACKING_SIGMA, "axis": 0},
  ),
  "tracking_lin_vel_y": RewardTermCfg(
    func=track_lin_vel_axis_filtered,
    weight=1.0,
    params={"command_name": "twist", "std": _TRACKING_SIGMA, "axis": 1},
  ),
  "tracking_ang_vel": RewardTermCfg(
    func=track_ang_vel_yaw_filtered,
    weight=0.5,
    params={"command_name": "twist", "std": _TRACKING_SIGMA},
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
  # ----------------------- regularization ----------------------- #
  "base_height": RewardTermCfg(
    func=base_height_penalty,
    weight=-20.0,
    params={"target_height": 0.68},
  ),
  "orientation": RewardTermCfg(
    func=orientation_penalty,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "lin_vel_z": RewardTermCfg(
    func=lin_vel_z_filtered_penalty,
    weight=-2.0,
    params={"command_name": "twist"},
  ),
  "ang_vel_xy": RewardTermCfg(
    func=body_angular_velocity_penalty,
    weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "torques": RewardTermCfg(
    func=torques_penalty,
    weight=-2e-4,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_PATTERNS)},
  ),
  "torque_tiredness": RewardTermCfg(
    func=torque_tiredness_penalty,
    weight=-1e-2,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_PATTERNS),
      "effort_limits": ACTUATED_EFFORT_LIMITS,
    },
  ),
  "power": RewardTermCfg(
    func=power_penalty,
    weight=-2e-3,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_PATTERNS)},
  ),
  "dof_vel": RewardTermCfg(
    func=dof_vel_penalty,
    weight=-1e-4,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
  "dof_acc": RewardTermCfg(
    func=dof_acc_penalty,
    weight=-1e-7,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
  "action_rate": RewardTermCfg(func=action_rate_l2, weight=-1.0),
  "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
  # ----------------------- foot shaping ----------------------- #
  "feet_slip": RewardTermCfg(
    func=feet_slip,
    weight=-0.1,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "sensor_name": "feet_ground_contact",
      "foot_body_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES)),
    },
  ),
  "feet_yaw_diff": RewardTermCfg(
    func=feet_yaw_diff_penalty,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "feet_yaw_mean": RewardTermCfg(
    func=feet_yaw_mean_penalty,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  # booster feet_roll (sum sq foot roll) ~ foot_orientation_penalty (roll+pitch).
  "feet_roll": RewardTermCfg(
    func=foot_orientation_penalty,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "feet_distance": RewardTermCfg(
    func=feet_distance_penalty,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "min_dist": 0.2,
    },
  ),
}
