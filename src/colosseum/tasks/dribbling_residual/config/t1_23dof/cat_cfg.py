import math
from typing import Dict

from mjlab.envs.mdp.terminations import bad_orientation, nan_detection, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.mdp.actions import DelayedJointPositionActionCfg
from colosseum.mdp.velocity_command import CurriculumVelocityCommandCfg
from colosseum.robots.t1_23dof.constants import ACTION_SCALE
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg
from colosseum.tasks.dribbling_residual.mdp.ball_twist_command import (
  BallTwistCommandCfg,
)
from colosseum.tasks.dribbling_residual.mdp.ball_velocity_command import (
  BallVelocityCommandCfg,
)

commands: Dict[str, CommandTermCfg] = {
  "twist": BallTwistCommandCfg(),
  "ball_vel": BallVelocityCommandCfg(),
  "gait_phase": GaitPhaseCommandCfg(
    gait_freq_range=(1.5, 2.0),
    gate_command_name="twist",
    gate_speed_threshold=0.05,
  ),
}

VELOCITY_ACTION_SCALE = {k: v for k, v in ACTION_SCALE.items()}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": DelayedJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=VELOCITY_ACTION_SCALE,
    use_default_offset=True,
    max_delay_steps=2,
  )
}

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
  "nan": TerminationTermCfg(func=nan_detection),
}
