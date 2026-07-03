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

from colosseum.mdp.ball_rewards import (
  ball_vel_angle_body,
  ball_vel_norm,
  ball_vel_tracking_body,
  foot_ball_contact,
  robot_ball_distance,
  robot_ball_yaw_body,
)
from colosseum.mdp.rewards import (
  arm_phase,
  base_height_penalty,
  dof_acc_penalty,
  dof_vel_penalty,
  feet_distance_penalty,
  feet_no_slip,
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
  torque_tiredness_penalty,
  torques_penalty,
)
from colosseum.robots.t1_23dof.constants import (
  BASE_BODY_NAME,
  FOOT_BODY_NAMES,
  FOOT_SITE_NAMES,
)
from colosseum.tasks.dribbling.mdp.rewards import (
  ball_target_progress,
  ball_target_reached,
)
from colosseum.tasks.dribbling_residual.mdp.terminations import ball_lost_penalty

LEG_JOINT_PATTERNS = (".*Hip.*", ".*Knee.*", ".*Ankle.*")
LOWER_BODY_JOINT_PATTERNS = (".*Hip.*", ".*Knee.*", ".*Ankle.*", "Waist")

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
  # ==============================
  # Ball / dribble task rewards
  # ==============================
  # Body-frame trackers (obstacle-free: no adversary in the residual task, so
  # the dribbling task's relaxed variants would just no-op).
  "ball_vel_tracking": RewardTermCfg(
    func=ball_vel_tracking_body,
    weight=3.0,
    params={"command_name": "ball_vel", "sharpness": 5.0},
  ),
  "ball_vel_norm": RewardTermCfg(
    func=ball_vel_norm,
    weight=2.0,
    params={"command_name": "ball_vel", "sharpness": 5.0},
  ),
  "ball_vel_angle": RewardTermCfg(
    func=ball_vel_angle_body,
    weight=1.0,
    params={"command_name": "ball_vel"},
  ),
  "robot_ball_distance": RewardTermCfg(
    func=robot_ball_distance,
    weight=0.01,
    params={
      "close_distance": 0.3,
      "behind_close_penalty": 0.5,
      "far_sharpness": 3.0,
      "between_feet_forward_distance": 0.1,
      "between_feet_penalty": 4.0,
    },
  ),
  # Ungated bootstrap: pays the instant a foot touches the ball, so the residual
  # has a gradient toward engagement before any ball-velocity reward can fire.
  "foot_ball_contact": RewardTermCfg(
    func=foot_ball_contact,
    weight=0.5,
    params={
      "sensor_name": "foot_ball_contact",
      "command_name": "ball_vel",
    },
  ),
  "robot_ball_yaw": RewardTermCfg(
    func=robot_ball_yaw_body,
    weight=2.0,
    params={"command_name": "ball_vel"},
  ),
  "ball_target_progress": RewardTermCfg(  # Obstacle gate auto-disables (no adversary term).
    func=ball_target_progress,
    weight=2.5,
    params={
      "command_name": "ball_vel",
      "target_near_distance": 0.15,
      "target_far_distance": 0.5,
      "speed_ref": 1.0,
      "distance_scale_ref": 2.0,
      "distance_scale_max": 1.5,
    },
  ),
  # Penalty for losing the ball (out of the head's FOV for > 3 s -> episode ends).
  # The target is a dribble-direction proxy with no arrival bonus, so the only
  # discrete event is this failure.
  "ball_lost": RewardTermCfg(
    func=ball_lost_penalty,
    weight=-10.0,
    params={"term_name": "ball_lost"},
  ),
  # =======================
  # Task Tracking Rewards
  # =======================
  "feet_swing": RewardTermCfg(
    func=feet_swing,
    weight=1.0,
    params={
      "phase_command_name": "gait_phase",
      "sensor_name": "feet_ground_contact",
      "swing_period": 0.2,
      "contact_threshold": 0.1,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "penalty_hip_pose": RewardTermCfg(
    func=pose_deviation_penalty,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*Hip_Roll",
          ".*Hip_Yaw",
          ".*Hip_Pitch",
          ".*Knee_Pitch",
          ".*Ankle_Roll",
        ),
      ),
      "weights_standing": {
        ".*Hip_Yaw": 5.0,
        ".*Hip_Roll": 2.0,
        ".*Hip_Pitch": 0.5,
        ".*Knee_Pitch": 0.5,
        ".*Ankle_Roll": 2.0,
      },
    },
  ),
  "feet_phase": RewardTermCfg(
    func=feet_phase,
    weight=1.0,
    params={
      "phase_command_name": "gait_phase",
      "height_sensor_name": "foot_height_scan",
      "swing_height": 0.15,
      "tracking_sigma": 0.001,
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
      "max_speed": 1.0,
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
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "penalty_orientation": RewardTermCfg(
    func=orientation_penalty,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "penalty_base_height": RewardTermCfg(
    func=base_height_penalty,
    weight=-30.0,
    params={"target_height": 0.64},
  ),
  # "penalty_feet_ori": RewardTermCfg(
  #   func=foot_orientation_penalty,
  #   weight=-1.0,
  #   params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  # ),
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
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
  "penalty_dof_acc": RewardTermCfg(
    func=dof_acc_penalty,
    weight=-1e-6,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
  # Effort penalties (booster_gym weights): price torque magnitude, proximity to
  # the torque limit, and positive mechanical power. Legs only (upper body is
  # PD-held and its holding torque is not under policy control).
  "penalty_torques": RewardTermCfg(
    func=torques_penalty,
    weight=-2e-5,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS)},
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
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERNS)},
  ),
}
