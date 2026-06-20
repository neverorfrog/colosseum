"""Reward configuration for the t1-kick AMP residual task.

The AMP discriminator (see ResidualAMPPPO + the ``amp`` obs group) now supplies
the *style* prior — what a natural kick/walk transition looks like — so the
hand-tuned gait/pose shaping stack is gone. What remains is split into two
buckets the discriminator cannot replace:

  TASK ("what to achieve") — the kick objective + the velocity trackers that
    steer the frozen walk toward the ball.
  SAFETY / sim2real regularization — actuator limits, joint vel/acc, torque,
    power, contact slip. AMP is blind to these (the reference clip can't teach
    actuator limits or MuJoCo contact slip), so they stay regardless of AMP, at
    low weights. Sourced from the actuators-branch velocity task (the newest,
    sim2real-tuned weights), including the torque/power effort stack.

Deliberately DROPPED (AMP now provides them): pose_deviation, feet_phase,
feet_swing, foot/feet orientation, feet_yaw_diff/mean, body_ang_vel, soft
landing, lateral velocity. ``penalty_orientation`` is kept only as a light
fall-safety floor (termination handles real falls).
"""

import math

from mjlab.envs.mdp import (
  action_rate_l2,
  is_alive,
  joint_pos_limits,
)
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  track_angular_velocity,
  track_linear_velocity,
)

from colosseum.mdp.rewards import (
  dof_acc_penalty,
  dof_vel_penalty,
  feet_distance_penalty,
  feet_slip,
  orientation_penalty,
  power_penalty,
  torque_tiredness_penalty,
  torques_penalty,
)
from colosseum.robots.t1_23dof.constants import (
  BASE_BODY_NAME,
  FOOT_BODY_NAMES,
  FOOT_SITE_NAMES,
)
from colosseum.tasks.kick.mdp.rewards import (
  ball_speed_kick_to_goal,
  ball_vel_angle_to_kick_goal,
  foot_ball_contact_motion,
  robot_ball_yaw_to_kick_goal,
)

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
  # TASK — kick objective
  # ==============================
  "ball_speed_kick": RewardTermCfg(
    func=ball_speed_kick_to_goal,
    weight=2.0,
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
  "ball_vel_angle": RewardTermCfg(
    func=ball_vel_angle_to_kick_goal,
    weight=2.0,
    params={"command_name": "ball_angle", "min_speed": 0.5},
  ),
  # Ungated bootstrap: pays the instant a foot sets the ball moving, so the
  # residual has a gradient toward engagement before any ball-velocity reward.
  "foot_ball_contact": RewardTermCfg(
    func=foot_ball_contact_motion,
    weight=1.0,
    params={"sensor_name": "foot_ball_contact"},
  ),
  "robot_ball_yaw": RewardTermCfg(
    func=robot_ball_yaw_to_kick_goal,
    weight=2.0,
    params={"command_name": "ball_angle"},
  ),
  # ==============================
  # TASK — velocity tracking (steers the frozen walk toward the ball).
  # mjlab built-ins (command-type-agnostic): the kick's `twist` is a
  # BallTwistCommand, which doesn't expose the EMA `filtered_lin_vel` the
  # actuators-velocity per-axis trackers need, so use the plain trackers here.
  # ==============================
  "track_linear_velocity": RewardTermCfg(
    func=track_linear_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  "track_angular_velocity": RewardTermCfg(
    func=track_angular_velocity,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.25)},
  ),
  "alive": RewardTermCfg(func=is_alive, weight=0.25),
  # =========================
  # SAFETY / sim2real regularization (AMP is blind to these).
  # =========================
  # Light fall-safety floor only (termination handles real falls); kept low so
  # it doesn't fight the torso lean the kick needs.
  "penalty_orientation": RewardTermCfg(
    func=orientation_penalty,
    weight=-2.0,
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
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERNS)
    },
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
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERNS)
    },
  ),
}
