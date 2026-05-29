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

commands: Dict[str, CommandTermCfg] = {
  "twist": CurriculumVelocityCommandCfg(
    entity_name="robot",
    rel_forward_envs=0.3,
    rel_heading_envs=0.4,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    resampling_time_range=(7.0, 10.0),
    ranges=CurriculumVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.5),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.5, 1.5),
      heading=(-math.pi, math.pi),
    ),
    # Performance-gated 2D grid. Lateral is coupled to the forward level and
    # capped below it (res_y < res_x), so "max forward AND max lateral" is never
    # commanded; cells only open up once an env tracks the frontier within
    # tolerance. Max |vx|=1.5, |vy|=0.6, |wz|=1.2 at the outermost levels.
    lin_levels=6,
    ang_levels=6,
    lin_vel_x_resolution=0.25,
    lin_vel_y_resolution=0.10,
    ang_vel_resolution=0.20,
    x_toler=0.30,
    y_toler=0.15,
    yaw_toler=0.20,
    update_rate=0.10,
  ),
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
