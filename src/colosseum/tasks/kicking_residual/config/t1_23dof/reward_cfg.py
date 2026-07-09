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
  dof_acc_penalty,
  dof_vel_penalty,
  feet_distance_penalty,
  feet_phase,
  feet_slip,
  lateral_velocity_penalty,
  orientation_penalty,
)
from colosseum.robots.t1.constants import (
  BASE_BODY_NAME,
  FOOT_BODY_NAMES,
  FOOT_SITE_NAMES,
)
from colosseum.tasks.dribbling.mdp.rewards import (
  ball_target_progress,
  ball_target_reached,
)
from colosseum.tasks.kicking_residual.mdp.rewards import (
  ball_speed_kick_to_goal,
  ball_vel_angle_to_kick_goal,
  foot_ball_contact_motion,
  robot_ball_yaw_to_kick_goal,
)

rewards = {
  # ==============================
  # Ball / dribble task rewards
  # ==============================
  "ball_speed_kick": RewardTermCfg(
    func=ball_speed_kick_to_goal,
    weight=3.0,
    params={
      "command_name": "ball_angle",
      "max_reward": 8.0,
      "saturation_velocity": 4.0,
      "foot_contact_sensor_name": "foot_ball_contact",
      "min_contact_force": 3.0,
      "kick_credit_steps": 50,
      "direction_weight": 1.0,
    },
  ),
  # Body-frame trackers (obstacle-free: no adversary in the residual task, so
  # the dribbling task's relaxed variants would just no-op).
  "ball_vel_angle": RewardTermCfg(
    func=ball_vel_angle_to_kick_goal,
    weight=3.0,
    params={
      "command_name": "ball_angle",
      "min_speed": 0.5,
    },
  ),
  # Ungated bootstrap: pays the instant a foot touches the ball, so the residual
  # has a gradient toward engagement before any ball-velocity reward can fire.
  "foot_ball_contact": RewardTermCfg(
    func=foot_ball_contact_motion,
    weight=1.0,
    params={
      "sensor_name": "foot_ball_contact",
    },
  ),
  "robot_ball_yaw": RewardTermCfg(
    func=robot_ball_yaw_to_kick_goal,
    weight=2.0,
    params={"command_name": "ball_angle"},
  ),
  # =======================
  # Gait shaping (locomotion is handled by the frozen walk, which already tracks
  # `twist`; no velocity-tracking reward here — it would just re-reward the base
  # skill and fight the plant-and-strike at the kick moment).
  # =======================
  "feet_phase": RewardTermCfg(
    func=feet_phase,
    weight=0.5,
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
    weight=0.5,
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
    weight=0.25,
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
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "penalty_lateral_vel": RewardTermCfg(
    func=lateral_velocity_penalty,
    weight=-1.0,
  ),
  "penalty_orientation": RewardTermCfg(
    func=orientation_penalty,
    weight=-5.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
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
  "feet_slip": RewardTermCfg(
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
    weight=-1e-4,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
  "penalty_dof_acc": RewardTermCfg(
    func=dof_acc_penalty,
    weight=-1e-7,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  ),
}
