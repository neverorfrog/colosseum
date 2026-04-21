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
from colosseum.tasks.dribbling.mdp.terminations import ball_captured, ball_lost
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
    speed_range=(0.2, 1.0),
    target_distance_range=(2.0, 3.0),
    speed_gain=1.0,
    target_reached_threshold=0.5,
    heading_range=math.pi / 4,  # ±45° around the robot forward direction
    resampling_time_range=(5.0, 10.0),
    debug_vis=True,
  ),
  "gait_phase": GaitPhaseCommandCfg(gait_freq_range=(1.5, 2.5)),
  # Obstacle command: starts with 0 active obstacles (unlocked by curriculum).
  "adversary": ObstacleCommandCfg(
    num_obstacles=NUM_OBSTACLES,
    num_active=0,
    behavior="none",
    distance_range=(1.5, 3.0),
    lateral_offset_range=(-0.8, 0.8),
    min_speed=0.0,
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
        {"step": 16_000, "half_range": math.pi / 6},  # ±30°
        {"step": 48_000, "half_range": math.pi / 3},  # ±60°
        {"step": 96_000, "half_range": math.pi / 2},  # ±90°
        {"step": 144_000, "half_range": math.pi},  # ±180° (full)
      ],
    },
  ),
  "push_ball": CurriculumTermCfg(
    func=push_ball_curriculum,
    params={
      "event_name": "push_ball",
      "stages": [
        {"step": 0, "max_speed": 0.3},
        {"step": 48_000, "max_speed": 0.6},
        {"step": 112_000, "max_speed": 1.0},
      ],
    },
  ),
  "obstacle": CurriculumTermCfg(
    func=obstacle_curriculum,
    params={
      "command_name": "adversary",
      # Thresholds are stretched for a long 2B-step run with the current
      # distributed setup. Curriculum uses common_step_counter, i.e.
      #   curriculum_step = global_step / num_envs_per_rank
      # With 10,240 envs per rank:
      # - 500M global steps  -> ~48.8k curriculum steps
      # - 2B   global steps  -> ~195.3k curriculum steps
      # so the 2B schedule is exactly 4x longer than the 500M one.
      #
      # The harder stages get more room on purpose:
      # - `static_blocker` is the first real obstacle-avoidance stage and
      #   needs a long window to preserve the no-obstacle dribbling behavior.
      # - `ball_attacker` is the hardest single-obstacle stage and also gets
      #   extra time before moving to the cluttered multi-obstacle setting.
      #
      # Approximate stage durations:
      # - none:            19.5k curriculum steps
      # - static_blocker:  48.9k
      # - lateral_blocker: 29.3k
      # - ball_attacker:   52.7k
      # - mixed_attackers: 44.9k
      "stages": [
        {
          "step": 0,
          "num_active": 0,
          "behavior": "none",
          "max_speed": 0.0,
        },
        {
          "step": 19_500,
          "num_active": 1,
          "behavior": "static_blocker",
          "lateral_offset_range": (-0.3, 0.3),
          "forward_fraction_range": (0.35, 0.75),
          "max_speed": 0.0,
        },
        {
          "step": 68_400,
          "num_active": 1,
          "behavior": "lateral_blocker",
          "lateral_offset_range": (-0.3, 0.3),
          "forward_fraction_range": (0.35, 0.75),
          "min_speed": 0.05,
          "max_speed": 0.15,
          "velocity_resample_time_range": (0.6, 1.2),
        },
        {
          "step": 97_700,
          "num_active": 1,
          "behavior": "ball_attacker",
          "lateral_offset_range": (-0.3, 0.3),
          "forward_fraction_range": (0.35, 0.75),
          "min_speed": 0.08,
          "max_speed": 0.22,
          "velocity_resample_time_range": (0.4, 0.9),
        },
        {
          "step": 150_400,
          "num_active": 3,
          "behavior": "mixed_attackers",
          # Distractors still use distance_range for their random-angle spawn.
          "distance_range": (2.0, 3.0),
          "lateral_offset_range": (-0.3, 0.3),
          "forward_fraction_range": (0.35, 0.75),
          "min_speed": 0.05,
          "max_speed": 0.22,
          "velocity_resample_time_range": (0.4, 1.0),
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
  # "ball_captured": TerminationTermCfg(
  #   func=ball_captured,
  #   params={"command_name": "adversary", "capture_radius": 0.25},
  # ),
  "ball_lost": TerminationTermCfg(
    func=ball_lost,
    params={"max_robot_ball_distance": 2.0},
  ),
}
