"""Commands, actions, curriculum, and terminations for T1 soccer-maze task."""

import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.robots.t1_23dof.constants import ACTION_SCALE
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg
from colosseum.tasks.maze.mdp.abstraction_velocity_command import (
  AbstractionVelocityCommandCfg,
)
from colosseum.tasks.maze.mdp.curriculums import (
  base_velocity_curriculum,
  wall_collision_termination_curriculum,
)
from colosseum.tasks.maze.mdp.goal_command import MazeGoalCommandCfg
from colosseum.tasks.maze.mdp.terminations import arrived_at_goal, collided_with_wall

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
# Arms are held at home pose; only leg joints are trained.
_ACTION_SCALE = {k: (0.0 if k in _ARM_JOINTS else v) for k, v in ACTION_SCALE.items()}

commands: Dict[str, CommandTermCfg] = {
  # Ball target velocity: world-frame direction from the maze abstraction
  # queried at the ball's current position.
  "ball_vel": AbstractionVelocityCommandCfg(
    query_entity="ball",
    omnidirectional=True,
    use_root_pos=False,  # ball has no root_site
    base_velocity=0.3,  # ramped up by curriculum
    ema_smoothing=0.2,
    abstraction_name="grid",
  ),
  "goal": MazeGoalCommandCfg(static_goals=True),
  "gait_phase": GaitPhaseCommandCfg(),
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=_ACTION_SCALE,
    use_default_offset=True,
  )
}

curriculum: dict[str, CurriculumTermCfg] = {
  "command_vel": CurriculumTermCfg(
    func=base_velocity_curriculum,
    params={
      "command_name": "ball_vel",
      "velocity_stages": [
        {"step": 0, "base_velocity": 0.3},
        {"step": 1000 * 24, "base_velocity": 0.6},
        {"step": 5000 * 24, "base_velocity": 1.0},
      ],
    },
  ),
  # "wall_termination": CurriculumTermCfg(
  #   func=wall_collision_termination_curriculum,
  #   params={
  #     "end_step": 500_000_000,
  #     "start_prob": 0.0,
  #     "end_prob": 1.0,
  #   },
  # ),
}

terminations: dict[str, TerminationTermCfg] = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
  "arrived_at_goal": TerminationTermCfg(
    func=arrived_at_goal,
    params={
      "asset_cfg": SceneEntityCfg("ball", site_names=("root_site",)),
      "threshold": 0.5,
    },
  ),
  # "wall_collision": TerminationTermCfg(
  #   func=collided_with_wall,
  #   params={
  #     "sensor_name": "wall_collision",
  #     "force_threshold": 100.0,
  #   },
  # ),
}
