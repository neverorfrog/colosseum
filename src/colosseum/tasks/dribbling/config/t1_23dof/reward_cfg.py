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
  robot_ball_approach_vel,
  robot_ball_distance,
  ball_vel_angle_body,
  ball_vel_norm,
  ball_vel_tracking_body,
  feet_distance_penalty,
  obstacle_avoidance,
  pose_deviation,
  robot_ball_yaw_body,
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
    weight=4.0,
    params={
      "command_name": "ball_vel",  # Which ball-speed command to match.
      "sharpness": 1.0,  # Larger -> tighter speed matching.
    },
  ),
  "ball_vel_angle": RewardTermCfg(  # Align ball-motion direction with the command.
    func=ball_vel_angle_body,
    weight=4.0,
    params={"command_name": "ball_vel"},  # Which ball-direction command to align with.
  ),
  "robot_ball_distance": RewardTermCfg(  # Keep the robot reasonably close to the ball.
    func=robot_ball_distance,
    weight=2.0,
    params={"sharpness": 1.0},  # Larger -> reward drops faster as robot-ball distance grows.
  ),
  "robot_ball_yaw": RewardTermCfg(  # Keep the ball in front of the robot along the command.
    func=robot_ball_yaw_body,
    weight=2.0,
    params={"command_name": "ball_vel"},  # Which command defines the preferred facing direction.
  ),
  "robot_ball_approach_vel": RewardTermCfg(  # Reward base motion that closes distance to the ball.
    func=robot_ball_approach_vel,
    weight=2.0,
    params={"command_name": "ball_vel"},  # Command whose speed sets the desired approach urgency.
  ),
  "obstacle_avoidance": RewardTermCfg(  # Local obstacle penalty when the nearest obstacle is near and in front.
    func=obstacle_avoidance,
    weight=-1.5,
    params={
      "command_name": "adversary",  # Obstacle command term providing obstacle positions/velocities.
      "ball_vel_command_name": "ball_vel",  # Ball command used for the kick-direction penalty.
      "collision_detection_range": 1.2,  # Body-obstacle penalty only inside this robot-obstacle distance.
      "direction_detection_range": 2.0,  # Kick-direction penalty only inside this robot-obstacle distance.
      "collision_near_distance": 0.5,  # Maximum collision penalty at or below this distance.
      "collision_far_distance": 1.2,  # Collision penalty fades to zero at or above this distance.
      "direction_sharpness": 4.0,  # Larger -> stronger penalty for commanding the ball toward the obstacle.
      "collision_weight": 0.35,  # Relative weight of body-obstacle proximity.
      "direction_weight": 1.0,  # Relative weight of "do not kick toward obstacle".
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
