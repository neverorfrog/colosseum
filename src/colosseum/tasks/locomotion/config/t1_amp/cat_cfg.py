import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, nan_detection, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.mdp.actions import HeadPerturbActionCfg
from colosseum.mdp.velocity_command import (
  CurriculumVelocityCommandCfg,
  TrueErrorVelocityCommandCfg,
)
from colosseum.robots.t1.constants import JOINT_NAMES, ACTION_SCALE
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg

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
    ranges=CurriculumVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-0.5, 0.5),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
    # # Performance-gated 2D grid. Lateral is coupled to the forward level and
    # # capped below it (res_y < res_x), so "max forward AND max lateral" is never
    # # commanded; cells only open up once an env tracks the frontier within
    # # tolerance. Max |vx|=1.375, |vy|=0.5, |wz|=1.1 at the outermost levels.
    # lin_levels=5,
    # ang_levels=5,
    # lin_vel_x_resolution=0.25,
    # lin_vel_y_resolution=0.10,
    # ang_vel_resolution=0.20,
    # # Gate mirrors t1.py: at episode reset, promote if the env survived
    # # >=(1-episode_length_toler) of the episode AND its EMA-filtered velocity
    # # (filter_weight) tracked the command within these tolerances. x_toler is
    # # kept below lin_vel_x_resolution so standing still cannot satisfy the
    # # gate at level >= 1 (the gate ANDs all axes, x is the binding one).
    # x_toler=0.15,
    # y_toler=0.20,
    # yaw_toler=0.20,
    # episode_length_toler=0.10,
    # filter_weight=0.10,
    # update_rate=0.10,
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
