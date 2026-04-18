import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.robots.t1_23dof.constants import ACTION_SCALE
from colosseum.tasks.dribbling.mdp.ball_velocity_command import BallVelocityCommandCfg
from colosseum.tasks.dribbling.mdp.curriculum import (
  obstacle_curriculum,
  push_ball_curriculum,
  yaw_reset_curriculum,
)
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg
from colosseum.tasks.dribbling.mdp.head_ik_action import HeadIKActionCfg
from colosseum.tasks.dribbling.mdp.obstacle_commands import ObstacleCommandCfg
from colosseum.tasks.dribbling.mdp.terminations import ball_captured
from colosseum.tasks.dribbling.obstacle_spec import NUM_OBSTACLES

_ARM_JOINTS = {
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
}
_HEAD_JOINTS = {"AAHead_yaw", "Head_pitch"}
# Arms and head scale=0: head targets are overridden by HeadIKActionCfg below.
_DRIBBLING_ACTION_SCALE = {
  k: (0.0 if k in _ARM_JOINTS or k in _HEAD_JOINTS else v)
  for k, v in ACTION_SCALE.items()
}

commands: Dict[str, CommandTermCfg] = {
  "ball_vel": BallVelocityCommandCfg(
    robot_entity="robot",
    ball_entity="ball",
    speed_range=(0.3, 2.0),
    heading_range=math.pi / 2,  # ±90° from forward
    resampling_time_range=(10.0, 20.0),
    debug_vis=True,
  ),
  "gait_phase": GaitPhaseCommandCfg(),
  # Obstacle command: starts with 0 active obstacles (unlocked by curriculum).
  "adversary": ObstacleCommandCfg(
    num_obstacles=NUM_OBSTACLES,
    num_active=0,
    distance_range=(2.5, 4.0),
    max_speed=0.0,
  ),
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=_DRIBBLING_ACTION_SCALE,
    use_default_offset=True,
  ),
  # IK head tracking: runs after joint_pos, overrides the frozen head targets
  # with analytically computed yaw/pitch to keep the camera on the ball.
  "head_ik": HeadIKActionCfg(),
}

curriculum = {
  "yaw_reset": CurriculumTermCfg(
    func=yaw_reset_curriculum,
    params={
      "event_name": "reset_base",
      "stages": [
        {"step": 0, "half_range": 0.0},  # always forward
        {"step": 2000, "half_range": math.pi / 6},  # ±30°
        {"step": 6000, "half_range": math.pi / 3},  # ±60°
        {"step": 12000, "half_range": math.pi / 2},  # ±90°
        {"step": 20000, "half_range": math.pi},  # ±180° (full)
      ],
    },
  ),
  "push_ball": CurriculumTermCfg(
    func=push_ball_curriculum,
    params={
      "event_name": "push_ball",
      "stages": [
        {"step": 0, "max_speed": 0.3},
        {"step": 5000, "max_speed": 0.6},
        {"step": 12000, "max_speed": 1.0},
      ],
    },
  ),
  "obstacle": CurriculumTermCfg(
    func=obstacle_curriculum,
    params={
      "command_name": "adversary",
      "stages": [
        {"step": 0, "num_active": 0, "distance_range": (2.5, 4.0), "max_speed": 0.0},
        {
          "step": 2_000,
          "num_active": 1,
          "distance_range": (2.0, 4.0),
          "max_speed": 0.1,
        },
        {
          "step": 5_000,
          "num_active": 1,
          "distance_range": (1.5, 3.5),
          "max_speed": 0.5,
        },
        {
          "step": 8_000,
          "num_active": 2,
          "distance_range": (2.0, 4.0),
          "max_speed": 0.3,
        },
        {
          "step": 10_000,
          "num_active": 2,
          "distance_range": (1.5, 3.5),
          "max_speed": 0.5,
        },
        {
          "step": 15000,
          "num_active": 3,
          "distance_range": (2.0, 4.0),
          "max_speed": 0.3,
        },
        {
          "step": 20000,
          "num_active": 3,
          "distance_range": (1.5, 3.5),
          "max_speed": 0.5,
        },
      ],
    },
  ),
}

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
  "ball_captured": TerminationTermCfg(
    func=ball_captured,
    params={"command_name": "adversary", "capture_radius": 0.5},
  ),
}
