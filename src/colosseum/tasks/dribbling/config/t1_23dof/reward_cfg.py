import math  # noqa: F401

from mjlab.envs.mdp import (
  action_rate_l2,
  joint_pos_limits,
)
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  angular_momentum_penalty,
  body_angular_velocity_penalty,
  feet_swing_height,
  self_collision_cost,
  soft_landing,
)

from colosseum.mdp.rewards import flat_orientation
from colosseum.robots.t1_23dof.constants import (
  BASE_BODY_NAME,
  FOOT_SITE_NAMES,
)
from colosseum.robots.t1_23dof.sensors import (
  FOOT_FOOT_CONTACT_SENSOR,
  NONFOOT_BALL_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
)
from colosseum.tasks.dribbling.mdp.rewards import (
  ball_obstacle_collision,
  ball_target_progress,
  robot_ball_approach_vel,
  robot_ball_distance,
  ball_vel_angle_body,
  ball_vel_norm,
  ball_vel_tracking_body,
  feet_distance_penalty,
  obstacle_direction,
  pose_deviation,
  robot_ball_yaw_body,
  robot_obstacle_collision,
  stance_phase_schedule,
  swing_phase_schedule,
)

rewards = {
  # ------------------------------------------------------------------ #
  # Task rewards                                                         #
  # ------------------------------------------------------------------ #
  "ball_vel_tracking": RewardTermCfg(  # Match the full commanded ball-velocity vector.
    func=ball_vel_tracking_body,
    weight=3.0,
    params={
      "command_name": "ball_vel",  # Which ball-velocity command to track.
      "sharpness": 1.5,  # Larger -> penalize vector tracking error more strongly.
    },
  ),
  "ball_vel_norm": RewardTermCfg(  # Match the commanded ball-speed magnitude.
    func=ball_vel_norm,
    weight=2.0,
    params={
      "command_name": "ball_vel",  # Which ball-speed command to match.
      "sharpness": 1.5,  # Larger -> tighter speed matching.
    },
  ),
  "ball_vel_angle": RewardTermCfg(  # Align ball-motion direction with the command.
    func=ball_vel_angle_body,
    weight=2.0,
    params={"command_name": "ball_vel"},  # Which ball-direction command to align with.
  ),
  "robot_ball_distance": RewardTermCfg(  # Keep the robot reasonably close to the ball.
    func=robot_ball_distance,
    weight=1.5,
    params={
      "close_distance": 0.3,  # Ball within 0.3 m and in front is considered fully controlled.
      "behind_close_penalty": 0.5,  # Constant penalty level when the ball is close but behind.
      "far_sharpness": 3.0,  # Larger -> stronger exponential decay once the ball is farther than 0.3 m.
    },
  ),
  "robot_ball_yaw": RewardTermCfg(  # Keep the ball in front of the robot along the command.
    func=robot_ball_yaw_body,
    weight=2.0,
    params={"command_name": "ball_vel"},  # Which command defines the preferred facing direction.
  ),
  "robot_ball_approach_vel": RewardTermCfg(  # Reward base motion that closes distance to the ball.
    func=robot_ball_approach_vel,
    weight=1.0,
    params={"command_name": "ball_vel"},  # Command whose speed sets the desired approach urgency.
  ),
  "ball_target_progress": RewardTermCfg(  # Reward moving the ball along the current ball-to-target direction.
    func=ball_target_progress,
    weight=2.0,
    params={
      "command_name": "ball_vel",  # Ball command term providing the persistent world-frame target.
      "obstacle_command_name": "adversary",  # Obstacle command term used to detect blocked target neighborhoods.
      "target_near_distance": 0.4,  # Progress reward is fully off when the ball is already this close to the target.
      "target_far_distance": 1.0,  # Progress reward ramps to full strength by this target distance.
      "target_obstacle_near_distance": 0.3,  # Progress reward is fully off when the target is this close to the nearest obstacle.
      "target_obstacle_far_distance": 0.8,  # Progress reward ramps back to full strength once the target is sufficiently far from obstacles.
      "speed_ref": 1.0,  # Ball speed toward the target that saturates the normalized progress reward.
    },
  ),
  "robot_obstacle_collision": RewardTermCfg(  # Local robot-obstacle safety term.
    func=robot_obstacle_collision,
    weight=-5.0,
    params={
      "command_name": "adversary",  # Obstacle command term providing obstacle positions/velocities.
      "collision_detection_range": 1.5,  # Collision penalty only inside this robot-obstacle distance.
      "collision_near_distance": 0.5,  # Maximum collision penalty at or below this distance.
      "collision_far_distance": 1.5,  # Collision penalty fades to zero at or above this distance.
    },
  ),
  "ball_obstacle_collision": RewardTermCfg(  # Penalize the ball physically touching an obstacle.
    func=ball_obstacle_collision,
    weight=-3.0,
    params={
      "command_name": "adversary",  # Obstacle command term providing obstacle positions/velocities.
      "collision_detection_range": 1.0,  # Ball-obstacle distance beyond which the penalty is zero.
      "collision_near_distance": 0.15,  # Maximum penalty at or below this ball-obstacle distance (≈ ball radius + obstacle radius).
      "collision_far_distance": 0.6,  # Penalty fades to zero at or above this ball-obstacle distance.
    },
  ),
  "obstacle_direction": RewardTermCfg(  # Penalize commanding the ball toward an obstacle that blocks the target path.
    func=obstacle_direction,
    weight=-2.5,
    params={
      "command_name": "adversary",  # Obstacle command term providing obstacle positions/velocities.
      "ball_vel_command_name": "ball_vel",  # Ball command term providing the persistent target.
      "direction_detection_range": 3.0,  # Only obstacles within this forward target-segment range affect direction.
      "direction_tube_radius": 1.0,  # Direction term only if the obstacle lies close to the ball-target segment.
      "direction_sharpness": 3.0,  # Larger -> sharper bounded penalty for commanding the ball toward the obstacle.
      "ball_engagement_near_distance": 0.3,  # Full obstacle pressure only when the ball is under close control.
      "ball_engagement_far_distance": 0.75,  # Obstacle pressure fades out when the ball is not engaged.
    },
  ),
  # ------------------------------------------------------------------ #
  # Locomotion regularization                                            #
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
      "command_name": "ball_vel",
      "command_threshold": 0.05,
    },
  ),
  "soft_landing": RewardTermCfg(
    func=soft_landing,
    weight=-1e-5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "ball_vel",
      "command_threshold": 0.05,
    },
  ),
  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-1.0,
    params={"sensor_name": SELF_COLLISION_SENSOR.name, "force_threshold": 10.0},
  ),
  "feet_distance": RewardTermCfg(
    func=feet_distance_penalty,
    weight=-4.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      "min_dist": 0.15,
    },
  ),
  "foot_foot_contact": RewardTermCfg(
    func=self_collision_cost,
    weight=-3.0,
    params={"sensor_name": FOOT_FOOT_CONTACT_SENSOR.name, "force_threshold": 1.0},
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
    weight=4.0,
    params={
      "phase_command_name": "gait_phase",
      "sensor_name": "feet_ground_contact",
    },
  ),
  "stance_phase": RewardTermCfg(
    func=stance_phase_schedule,
    weight=4.0,
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
        joint_names=(
          r"(?i).*shoulder.*",
          r"(?i).*elbow.*",
        ),
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
        joint_names=(
          r"(?i).*hip.*",
          r"(?i).*knee.*",
          r"(?i).*ankle.*",
        ),
      ),
      "std": 0.3,
    },
  ),
}
