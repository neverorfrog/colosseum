import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.robots.t1_23dof.constants import ACTION_SCALE
from colosseum.tasks.dribbling.mdp.ball_velocity_command import BallVelocityCommandCfg

commands: Dict[str, CommandTermCfg] = {
  "ball_vel": BallVelocityCommandCfg(
    robot_entity="robot",
    ball_entity="ball",
    speed_range=(0.3, 1.0),
    heading_range=math.pi / 3,
    resampling_time_range=(20.0, 30.0),
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

curriculum = {}

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
}
