"""Commands, actions, curriculum, and terminations for T1 maze task."""

import math

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.robots.t1_23dof.constants import LOCOMOTION_ACTION_SCALE
from colosseum.mdp.abstraction.maze.abstraction_velocity_command import (
  AbstractionVelocityCommandCfg,
)
from colosseum.tasks.maze.mdp.curriculums import (
  base_velocity_curriculum,
  wall_collision_termination_curriculum,
)
from colosseum.mdp.abstraction.maze.goal_command import MazeGoalCommandCfg
from colosseum.tasks.maze.mdp.terminations import arrived_at_goal, collided_with_wall

commands: dict[str, CommandTermCfg] = {
  "goal": MazeGoalCommandCfg(static_goals=True),
  "velocity": AbstractionVelocityCommandCfg(
    body_forward_axis=(1.0, 0.0, 0.0),
    angular_velocity_gain=0.5,
    ema_smoothing=0.1,
    rel_standing_envs=0.1,
  ),
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=LOCOMOTION_ACTION_SCALE,
    use_default_offset=True,
  )
}

curriculum: dict[str, CurriculumTermCfg] = {
  "command_vel": CurriculumTermCfg(
    func=base_velocity_curriculum,
    params={
      "command_name": "velocity",
      "velocity_stages": [
        {"step": 0, "base_velocity": 0.1},
        {"step": 1000 * 24, "base_velocity": 0.4},
        {"step": 5000 * 24, "base_velocity": 0.7},
        {"step": 10000 * 24, "base_velocity": 1.0},
      ],
    },
  ),
  "wall_termination": CurriculumTermCfg(
    func=wall_collision_termination_curriculum,
    params={
      "end_step": 500_000_000,
      "start_prob": 0.0,
      "end_prob": 1.0,
    },
  ),
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
      "asset_cfg": SceneEntityCfg("robot", site_names=("root_site",)),
      "threshold": 0.5,
    },
  ),
  "wall_collision": TerminationTermCfg(
    func=collided_with_wall,
    params={
      "sensor_name": "wall_collision",
      "force_threshold": 100.0,
    },
  ),
}
