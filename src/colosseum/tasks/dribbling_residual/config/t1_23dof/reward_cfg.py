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
  dof_acc_penalty,
  dof_vel_penalty,
  feet_distance_penalty,
  feet_no_slip,
  feet_phase,
  feet_slip,
  feet_yaw_diff_penalty,
  foot_orientation_penalty,
  orientation_penalty,
  pose_deviation_penalty,
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

rewards = {
  # ==============================
  # Ball / dribble task rewards
  # ==============================
  # Body-frame trackers (obstacle-free: no adversary in the residual task, so
  # the dribbling task's relaxed variants would just no-op).
  "ball_vel_tracking": RewardTermCfg(
    func=ball_vel_tracking_body,
    weight=3.0,
    params={"command_name": "ball_vel", "sharpness": 1.5},
  ),
  "ball_vel_norm": RewardTermCfg(
    func=ball_vel_norm,
    weight=2.0,
    params={"command_name": "ball_vel", "sharpness": 1.5},
  ),
  "ball_vel_angle": RewardTermCfg(
    func=ball_vel_angle_body,
    weight=2.0,
    params={"command_name": "ball_vel"},
  ),
  "robot_ball_distance": RewardTermCfg(
    func=robot_ball_distance,
    weight=1.0,
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
    weight=1.0,
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
    weight=2.0,
    params={
      "command_name": "ball_vel",
      "target_near_distance": 0.15,
      "target_far_distance": 0.5,
      "speed_ref": 1.0,
      "distance_scale_ref": 2.0,
      "distance_scale_max": 1.5,
    },
  ),
  "ball_target_reached": RewardTermCfg(  # Reduced from dribbling's 50.0 for stabler early residual learning.
    func=ball_target_reached,
    weight=10.0,
    params={"command_name": "ball_vel"},
  ),
  # =======================
  # Task Tracking Rewards
  # =======================
  "feet_phase": RewardTermCfg(
    func=feet_phase,
    weight=1.0,
    params={
      "phase_command_name": "gait_phase",
      "height_sensor_name": "foot_height_scan",
      "swing_height": 0.08,
      "tracking_sigma": 0.005,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
  "arm_phase": RewardTermCfg(
    func=arm_phase,
    weight=1.0,
    params={
      "phase_command_name": "gait_phase",
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("Left_Shoulder_Pitch", "Right_Shoulder_Pitch"),
      ),
      "swing_amplitude": 0.25,
      "max_speed": 1.5,
      "tracking_sigma": 0.25,
      "command_name": "twist",
    },
  ),
  "alive": RewardTermCfg(
    func=is_alive,
    weight=0.5,
  ),
  # =========================
  # Regularization penalties
  # =========================
  "penalty_landing": RewardTermCfg(
    func=soft_landing,
    weight=-0.005,
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
    weight=-10.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "penalty_feet_ori": RewardTermCfg(
    func=foot_orientation_penalty,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "penalty_feet_yaw_diff": RewardTermCfg(
    func=feet_yaw_diff_penalty,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
  ),
  "penalty_action_rate": RewardTermCfg(func=action_rate_l2, weight=-1.0),
  "penalty_pose_deviation": RewardTermCfg(
    func=pose_deviation_penalty,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "command_name": "twist",
      "walking_threshold": 0.05,
      "running_threshold": 1.5,
      "weights_standing": {},
      "weights_walking": {},
      "weights_running": {},
    },
  ),
  "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
  "penalty_feet_distance": RewardTermCfg(
    func=feet_distance_penalty,
    weight=-5.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "min_dist": 0.2,
    },
  ),
  "feet_slip": RewardTermCfg(
    func=feet_slip,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "sensor_name": "feet_ground_contact",
      "foot_body_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES)),
    },
  ),
  # Anchored L1 no-slip: catches slow standing creep (~1e-5 m/s) that the L2
  # feet_slip is blind to. Weight needs tuning — start small and raise until the
  # standing foot-creep disappears without hurting walking foot roll.
  "feet_no_slip": RewardTermCfg(
    func=feet_no_slip,
    weight=-20.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
      "sensor_name": "feet_ground_contact",
    },
  ),
  "penalty_dof_vel": RewardTermCfg(
    func=dof_vel_penalty,
    weight=-1e-4,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
  "penalty_dof_acc": RewardTermCfg(
    func=dof_acc_penalty,
    weight=-1e-7,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
}

# Standing: upper body tight + moderate leg constraint to prevent stance spread.
rewards["penalty_pose_deviation"].params["weights_standing"] = {
  r"(?i).*head.*": 50.0,
  r"(?i).*shoulder_pitch.*": 1.0,
  r"(?i).*shoulder_roll.*": 50.0,
  r"(?i).*elbow.pitch": 50.0,
  r"(?i).*elbow.yaw": 1.0,
  r"Waist": 25.0,
  r"(?i).*hip_pitch.*": 1.0,
  r"(?i).*hip_roll.*": 5.0,
  r"(?i).*hip_yaw.*": 5.0,
  r"(?i).*knee.*": 1.0,
  r"(?i).*ankle.*": 5.0,
}
# Walking: upper body tight (shoulder_pitch loose for arm_phase)
rewards["penalty_pose_deviation"].params["weights_walking"] = {
  r"(?i).*head.*": 50.0,
  r"(?i).*shoulder_pitch.*": 1.0,
  r"(?i).*shoulder_roll.*": 50.0,
  r"(?i).*elbow.pitch": 50.0,
  r"(?i).*elbow.yaw": 1.0,
  r"Waist": 15.0,
  r"(?i).*hip_pitch.*": 1.0,
  r"(?i).*hip_roll.*": 5.0,
  r"(?i).*hip_yaw.*": 5.0,
  r"(?i).*knee.*": 1.0,
  r"(?i).*ankle.*": 5.0,
}
# Running: same as walking — legs already nearly unconstrained.
rewards["penalty_pose_deviation"].params["weights_running"] = {
  r"(?i).*head.*": 50.0,
  r"(?i).*shoulder_pitch.*": 1.0,
  r"(?i).*shoulder_roll.*": 50.0,
  r"(?i).*elbow.pitch": 50.0,
  r"(?i).*elbow.yaw": 1.0,
  r"Waist": 5.0,
  r"(?i).*hip_pitch.*": 0.1,
  r"(?i).*hip_roll.*": 0.5,
  r"(?i).*hip_yaw.*": 0.5,
  r"(?i).*knee.*": 0.1,
  r"(?i).*ankle.*": 0.5,
}
