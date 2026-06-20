"""Reward configuration for t1-kicking-residual-mimic.

Reuses the ``t1_23dof`` residual task's full reward set unchanged (locomotion
+ kick rewards), and adds 6 motion-tracking terms that reward following the
reference kick clip (``models/trajectories/t1_motion.npz``). Each term is
wrapped in ``gated_motion_tracking_error``, which zeroes the reward until
``GatedHoldMotionCommand.triggered`` — i.e. before the robot is within 0.25 m
of the ball, the reference (a kick-ready stance) contributes nothing and
can't fight the walking gait.
"""

from mjlab.tasks.tracking.mdp import (
  motion_global_anchor_orientation_error_exp,
  motion_global_anchor_position_error_exp,
  motion_global_body_angular_velocity_error_exp,
  motion_global_body_linear_velocity_error_exp,
  motion_relative_body_orientation_error_exp,
  motion_relative_body_position_error_exp,
)
from mjlab.managers import RewardTermCfg

from colosseum.mdp.rewards import gated_motion_tracking_error

from ..t1_23dof.reward_cfg import rewards as _base_rewards

from colosseum.tasks.kicking_5.mdp.rewards import head_height_reward
from mjlab.managers.scene_entity_config import SceneEntityCfg
from colosseum.robots.t1_23dof.constants import HEAD_BODY_NAME
from colosseum.tasks.dribbling.mdp.rewards import robot_ball_distance


rewards = {
  **_base_rewards,
  "head_height": RewardTermCfg(
    func=head_height_reward,
    weight=1.5,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(HEAD_BODY_NAME,)),
      "target_height": 1.1,
      "std": 0.15,
    },
  ),
  "robot_ball_distance": RewardTermCfg(  # Keep the robot reasonably close to the ball.
    func=robot_ball_distance,
    weight=0.1,
    params={
      "close_distance": 0.3,  # Ball within 0.3 m and in front is considered fully controlled.
      "behind_close_penalty": 0.5,  # Constant penalty level when the ball is close but behind.
      "far_sharpness": 3.0,  # Larger -> stronger exponential decay once the ball is farther than 0.3 m.
      "between_feet_forward_distance": 0.1,  # |x_body| below this flags the ball as between the feet.
      "between_feet_penalty": 2.0,  # Larger -> stronger penalty when the ball ends up under the robot.
    },
  ),


  # ================================================================== #
  # Motion Tracking (gated: zero until within trigger_distance of ball) #
  # ================================================================== #
  "motion_global_anchor_pos": RewardTermCfg(
    func=gated_motion_tracking_error,
    weight=0.5,
    params={
      "command_name": "motion",
      "reward_fn": motion_global_anchor_position_error_exp,
      "std": 0.3,
    },
  ),
  "motion_global_anchor_ori": RewardTermCfg(
    func=gated_motion_tracking_error,
    weight=0.5,
    params={
      "command_name": "motion",
      "reward_fn": motion_global_anchor_orientation_error_exp,
      "std": 0.4,
    },
  ),
  "motion_body_pos": RewardTermCfg(
    func=gated_motion_tracking_error,
    weight=0.5,
    params={
      "command_name": "motion",
      "reward_fn": motion_relative_body_position_error_exp,
      "std": 0.3,
    },
  ),
  "motion_body_ori": RewardTermCfg(
    func=gated_motion_tracking_error,
    weight=0.5,
    params={
      "command_name": "motion",
      "reward_fn": motion_relative_body_orientation_error_exp,
      "std": 0.4,
    },
  ),
  "motion_body_lin_vel": RewardTermCfg(
    func=gated_motion_tracking_error,
    weight=0.5,
    params={
      "command_name": "motion",
      "reward_fn": motion_global_body_linear_velocity_error_exp,
      "std": 1.0,
    },
  ),
  "motion_body_ang_vel": RewardTermCfg(
    func=gated_motion_tracking_error,
    weight=0.5,
    params={
      "command_name": "motion",
      "reward_fn": motion_global_body_angular_velocity_error_exp,
      "std": 3.14,
    },
  ),
}
