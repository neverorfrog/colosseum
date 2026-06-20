"""Reward configuration for the t1-velocity-amp task (BeyondAMP-style walking).

Vanilla PPO learns to walk from scratch while an AMP discriminator (see AmpPPO +
the ``amp`` obs group) supplies the *style* — what a natural walking transition
looks like — so the hand-tuned gait/pose shaping stack from the base velocity
task is gone. What remains is split into two buckets the discriminator cannot
replace:

  TASK ("what to achieve") — the per-axis velocity trackers + alive bonus.
  SAFETY / sim2real regularization — actuator limits, joint vel/acc, torque,
    power, contact slip, foot spacing, a light orientation floor. AMP is blind
    to these (its obs is joint_pos/vel + base lin/ang vel only — no orientation,
    base height, or foot contacts; and the reference clip can't teach actuator
    limits or MuJoCo contact slip), so they stay regardless of AMP.

Deliberately DROPPED (AMP now provides them): feet_swing, feet_phase, arm_phase,
soft landing, body_ang_vel, lin_vel_z, base_height, foot orientation,
feet_yaw_diff/mean, pose_deviation. ``penalty_orientation`` is kept only as a
light fall-safety floor (termination handles real falls).
"""

import math

from mjlab.envs.mdp import (
  action_rate_l2,
  is_alive,
  joint_pos_limits,
)
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.mdp.rewards import (
  base_height_penalty,
  dof_acc_penalty,
  dof_vel_penalty,
  feet_distance_penalty,
  feet_phase,
  feet_slip,
  feet_swing,
  orientation_penalty,
  power_penalty,
  torque_tiredness_penalty,
  torques_penalty,
  track_ang_vel_yaw_filtered,
  track_linear_velocity_filtered,
)
from colosseum.robots.t1_23dof.constants import (
  BASE_BODY_NAME,
  FOOT_BODY_NAMES,
  FOOT_SITE_NAMES,
)

LEG_JOINT_PATTERNS = (".*Hip.*", ".*Knee.*", ".*Ankle.*")
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
  # TASK — velocity tracking
  # =======================
  # Combined xy kernel (mjlab stock form): a single Gaussian over both linear
  # axes, so an uncommanded axis (usually y) can't pay full reward for standing
  # still. Per-axis splitting let the y-tracker hand out ~3.0 for zero motion,
  # which created a standing optimum.
  "track_lin_vel": RewardTermCfg(
    func=track_linear_velocity_filtered,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  "track_ang_vel_yaw": RewardTermCfg(
    func=track_ang_vel_yaw_filtered,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  "feet_swing": RewardTermCfg(
    func=feet_swing,
    weight=2.0,
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
    weight=1.0,
    params={
      "phase_command_name": "gait_phase",
      "height_sensor_name": "foot_height_scan",
      "swing_height": 0.09,
      "tracking_sigma": 0.005,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "alive": RewardTermCfg(func=is_alive, weight=0.25),
  # =========================
  # SAFETY / sim2real regularization (AMP is blind to these).
  # =========================
  # Light fall-safety floor only (termination handles real falls); kept low so
  # it doesn't fight the natural torso motion AMP supplies.
  "penalty_orientation": RewardTermCfg(
    func=orientation_penalty,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "penalty_action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.1),
  "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
  "penalty_feet_distance": RewardTermCfg(
    func=feet_distance_penalty,
    weight=-1.0,
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
  # the torque limit, and positive mechanical power.
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
