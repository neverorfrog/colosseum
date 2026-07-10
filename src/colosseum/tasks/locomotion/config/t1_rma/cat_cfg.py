import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, nan_detection, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from colosseum.mdp.actions import HeadPerturbActionCfg
from colosseum.tasks.locomotion.mdp.velocity_command import TrueErrorVelocityCommandCfg
from colosseum.robots.t1.constants import ACTION_SCALE
from colosseum.tasks.locomotion.mdp.gait_phase_command import GaitPhaseCommandCfg

commands: Dict[str, CommandTermCfg] = {
  "twist": TrueErrorVelocityCommandCfg(
    entity_name="robot",
    rel_forward_envs=0.3,
    rel_heading_envs=0.3,
    rel_standing_envs=0.1,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    resampling_time_range=(7.0, 10.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.5, 1.5),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  ),
  "gait_phase": GaitPhaseCommandCfg(
    gait_freq_range=(1.5, 2.0),
    gate_command_name="twist",
    gate_speed_threshold=0.05,
  ),
}

FIXED_JOINTS = {}
UNACTUATED_JOINTS = {"AAHead_yaw", "Head_pitch"}
ACTION_SCALE: dict[str, float] = {
  name: (0.0 if name in FIXED_JOINTS else value)
  for name, value in ACTION_SCALE.items()
  if name not in UNACTUATED_JOINTS
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=("^(?!AAHead_yaw$|Head_pitch$).*$",),
    scale=ACTION_SCALE,
    use_default_offset=True,
  ),
  "head_perturb": HeadPerturbActionCfg(
    entity_name="robot",
    head_joint_names=("AAHead_yaw", "Head_pitch"),
    yaw_range=(-1.0, 1.0),
    pitch_range=(-0.2, 0.8),
    interval_range=(1.0, 3.0),
    smoothing_duration=0.3,
  ),
}

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
  "nan": TerminationTermCfg(func=nan_detection),
}
