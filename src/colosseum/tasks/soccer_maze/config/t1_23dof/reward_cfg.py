"""Reward configuration for T1 soccer-maze task."""

import math
from time import time

from mjlab.envs.mdp import action_rate_l2, joint_pos_limits
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  angular_momentum_penalty,
  body_angular_velocity_penalty,
  feet_slip,
  feet_swing_height,
  self_collision_cost,
  soft_landing,
)

from colosseum.mdp.rewards import flat_orientation
from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_SITE_NAMES
from colosseum.robots.t1_23dof.sensors import (
  FOOT_FOOT_CONTACT_SENSOR,
  NONFOOT_BALL_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
)
from colosseum.tasks.dribbling.mdp.rewards import (
  feet_distance_penalty,
  pose_deviation,
  stance_phase_schedule,
  swing_phase_schedule,
)
from colosseum.tasks.maze.mdp.rewards import wall_collisions
from colosseum.tasks.soccer_maze.mdp.rewards import (
  action_step_timeout_penalty,
  ball_at_final_goal,
  ball_push_target_progress,
  ball_push_target_reached,
  ball_vel_angle_body,
  ball_vel_norm_body,
  ball_vel_tracking_body,
  robot_ang_vel_tracking,
  robot_ball_approach_vel_push,
  robot_ball_distance_push,
  robot_ball_yaw_body,
  robot_heading_alignment,
  robot_lin_vel_tracking,
  time_penalty,
)

rewards = {
  # ------------------------------------------------------------------ #
  # Task: ball to target tracking (PUSH only)                           #
  # ------------------------------------------------------------------ #
  "ball_vel_tracking": RewardTermCfg(
    func=ball_vel_tracking_body,
    weight=3.0,
    params={"command_name": "sokoban", "sharpness": 1.0},
  ),
  "ball_vel_norm": RewardTermCfg(
    func=ball_vel_norm_body,
    weight=2.0,
    params={"command_name": "sokoban", "sharpness": 1.0},
  ),
  "ball_vel_angle": RewardTermCfg(
    func=ball_vel_angle_body,
    weight=3.0,
    params={"command_name": "sokoban"},
  ),
  "ball_push_target_progress": RewardTermCfg(
    func=ball_push_target_progress,
    weight=1.0,
    params={
      "command_name": "sokoban",
      "speed_ref": 0.5,
      "target_near_distance": 0.3,
      "target_far_distance": 0.7,
      "distance_scale_ref": 2.0,
      "distance_scale_max": 1.5,
    },
  ),
  "ball_push_target_reached": RewardTermCfg(
    func=ball_push_target_reached,
    weight=10.0,
    params={"command_name": "sokoban", "threshold": 0.5},
  ),
  "ball_at_final_goal": RewardTermCfg(
    func=ball_at_final_goal,
    weight=100.0,
    params={"command_name": "goal", "threshold": 0.5},
  ),
  "time_penalty": RewardTermCfg(
    func=time_penalty,
    weight=1.0,
  ),
  # ------------------------------------------------------------------ #
  # Progress: penalise stalling on a single plan step                   #
  # ------------------------------------------------------------------ #
  "action_step_timeout": RewardTermCfg(
    func=action_step_timeout_penalty,
    weight=0.1,
    params={"abstraction_name": "sokoban", "window_steps": 200},
  ),
  # ------------------------------------------------------------------ #
  # Task: robot velocity tracking (MOVE actions)                        #
  # ------------------------------------------------------------------ #
  "robot_lin_vel": RewardTermCfg(
    func=robot_lin_vel_tracking,
    weight=2.0,
    params={"command_name": "sokoban", "std": math.sqrt(0.25)},
  ),
  "robot_heading": RewardTermCfg(
    func=robot_heading_alignment,
    weight=1.0,
    params={"command_name": "sokoban"},
  ),
  # ------------------------------------------------------------------ #
  # Task: robot–ball relationship                                        #
  # ------------------------------------------------------------------ #
  "robot_ball_distance": RewardTermCfg(
    func=robot_ball_distance_push,
    weight=0.05,
    params={"command_name": "sokoban", "sharpness": 0.5},
  ),
  "robot_ball_yaw": RewardTermCfg(
    func=robot_ball_yaw_body,
    weight=0.5,
    params={"command_name": "sokoban"},
  ),
  "robot_ball_approach_vel": RewardTermCfg(
    func=robot_ball_approach_vel_push,
    weight=1.0,
    params={"command_name": "sokoban"},
  ),
  # ------------------------------------------------------------------ #
  # Maze: wall collision penalty                                         #
  # ------------------------------------------------------------------ #
  "wall_collisions": RewardTermCfg(
    func=wall_collisions,
    weight=-2.0,
    params={
      "sensor_name": "wall_collision",
      "min_robot_height": 0.3,
    },
  ),
  # ------------------------------------------------------------------ #
  # Locomotion regularization                                          #
  # ------------------------------------------------------------------ #
  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=1.0,
    params={
      "std": math.sqrt(0.2),
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME)),
    },
  ),
  "body_ang_vel": RewardTermCfg(
    func=body_angular_velocity_penalty,
    weight=-0.05,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
  ),
  "angular_momentum": RewardTermCfg(
    func=angular_momentum_penalty,
    weight=-0.5,
    params={"sensor_name": "robot/root_angmom"},
  ),
  "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
  "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.1),
  "foot_swing_height": RewardTermCfg(
    func=feet_swing_height,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.1,
      "command_name": "sokoban",
      "command_threshold": 0.05,
    },
  ),
  "foot_slip": RewardTermCfg(
    func=feet_slip,
    weight=-0.1,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "sokoban",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  ),
  "soft_landing": RewardTermCfg(
    func=soft_landing,
    weight=-1e-5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "sokoban",
      "command_threshold": 0.05,
    },
  ),
  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-1.0,
    params={"sensor_name": SELF_COLLISION_SENSOR.name, "force_threshold": 10.0},
  ),
  "foot_foot_contact": RewardTermCfg(
    func=self_collision_cost,
    weight=-3.0,
    params={"sensor_name": FOOT_FOOT_CONTACT_SENSOR.name, "force_threshold": 1.0},
  ),
  "feet_distance": RewardTermCfg(
    func=feet_distance_penalty,
    weight=-6.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_dist": 0.15,
    },
  ),
  "nonfoot_ball_contact": RewardTermCfg(
    func=self_collision_cost,
    weight=-2.0,
    params={"sensor_name": NONFOOT_BALL_CONTACT_SENSOR.name, "force_threshold": 1.0},
  ),
  # ------------------------------------------------------------------ #
  # Phase-schedule feet rewards                                          #
  # ------------------------------------------------------------------ #
  "swing_phase": RewardTermCfg(
    func=swing_phase_schedule,
    weight=3.0,
    params={
      "phase_command_name": "gait_phase",
      "sensor_name": "feet_ground_contact",
    },
  ),
  "stance_phase": RewardTermCfg(
    func=stance_phase_schedule,
    weight=2.0,
    params={
      "phase_command_name": "gait_phase",
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  ),
  # ------------------------------------------------------------------ #
  # Pose deviation                                                       #
  # ------------------------------------------------------------------ #
  "pose_arms": RewardTermCfg(
    func=pose_deviation,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(r"(?i).*shoulder.*", r"(?i).*elbow.*"),
      ),
      "std": 0.1,
    },
  ),
  "pose_legs": RewardTermCfg(
    func=pose_deviation,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(r"(?i).*hip.*", r"(?i).*knee.*", r"(?i).*ankle.*"),
      ),
      "std": 0.3,
    },
  ),
}
