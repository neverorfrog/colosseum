import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel

from colosseum.robots.t1_23dof.constants import ACTION_SCALE
from colosseum.tasks.dribbling.mdp.ball_approach_command import BallApproachCommandCfg

commands: Dict[str, CommandTermCfg] = {
  "ball_approach": BallApproachCommandCfg(
    robot_entity="robot",
    ball_entity="ball",
    base_velocity=0.8,
    min_velocity=0.1,
    max_velocity=1.5,
    ema_smoothing=0.2,
    body_forward_axis=(1.0, 0.0, 0.0),
    angular_velocity_gain=2.0,
    max_angular_velocity=1.5,
    min_alignment_scale=0.1,
    debug_vis=True,
  ),
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=ACTION_SCALE,
    use_default_offset=True,
  )
}

curriculum = {
  "command_vel": CurriculumTermCfg(
    func=commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
        {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
        {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
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
}
