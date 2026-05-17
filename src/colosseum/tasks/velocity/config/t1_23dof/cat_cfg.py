import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, nan_detection, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.mdp.velocity_command import TrueErrorVelocityCommandCfg
from colosseum.robots.t1_23dof.constants import ACTION_SCALE
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg

commands: Dict[str, CommandTermCfg] = {
  "twist": TrueErrorVelocityCommandCfg(
    entity_name="robot",
    rel_forward_envs=0.3,
    rel_heading_envs=0.5,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    resampling_time_range=(7.0, 10.0),
    ranges=TrueErrorVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.5, 1.0),
      lin_vel_y=(-0.8, 0.8),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  ),
  "gait_phase": GaitPhaseCommandCfg(
    gait_freq_range=(1.0, 1.5),
    gate_command_name="twist",
    gate_speed_threshold=0.05,
  ),
}

VELOCITY_ACTION_SCALE = {k: v for k, v in ACTION_SCALE.items()}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=VELOCITY_ACTION_SCALE,
    use_default_offset=True,
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
