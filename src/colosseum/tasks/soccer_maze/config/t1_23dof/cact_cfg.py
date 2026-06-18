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

from colosseum.mdp.abstraction.maze.goal_command import MazeGoalCommandCfg
from colosseum.robots.t1_23dof.constants import JOINT_NAMES
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg
from colosseum.tasks.maze.mdp.terminations import arrived_at_goal
from colosseum.tasks.soccer_maze.mdp.curriculums import sokoban_cache_curriculum
from colosseum.tasks.soccer_maze.mdp.sokoban_command import SokobanCommandCfg
from colosseum.tasks.soccer_maze.mdp.terminations import sokoban_plan_deviated

UPPER_BODY_JOINTS = {
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  "AAHead_yaw", 
  "Head_pitch"
}
ACTION_SCALE: dict[str, float] = {
  name: (0.0 if name in UPPER_BODY_JOINTS else 0.25) for name in JOINT_NAMES
}

commands: Dict[str, CommandTermCfg] = {
  "sokoban": SokobanCommandCfg(
    abstraction_name="sokoban",
    ball_speed=0.5,
    robot_speed=1.0,
    push_robot_speed=0.5,
  ),
  "goal": MazeGoalCommandCfg(static_goals=True),
  "gait_phase": GaitPhaseCommandCfg(),
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=ACTION_SCALE,
    use_default_offset=True,
  )
}

curriculum: dict[str, CurriculumTermCfg] = {
  "sokoban_cache": CurriculumTermCfg(
    func=sokoban_cache_curriculum,
    params={
      "abstraction_name": "sokoban",
      "clear_at_steps": [],
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
  "sokoban_deviated": TerminationTermCfg(
    func=sokoban_plan_deviated,
    params={"abstraction_name": "sokoban"},
  ),
  "arrived_at_goal": TerminationTermCfg(
    func=arrived_at_goal,
    params={
      "asset_cfg": SceneEntityCfg("ball", site_names=("root_site",)),
      "threshold": 0.6,
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
