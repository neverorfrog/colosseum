"""Actions / commands / terminations for the booster-mimic Waist+legs task.

The policy action space is Waist + 12 legs (13 DOF, scale 1.0). The model and
observations span the full 23-DOF body: arms stay pinned at their default pose
by their PD actuators, and the head is driven by ``head_perturb`` (randomly
perturbed during training, held at default in play).
"""

from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import (
  nan_detection,
  root_height_below_minimum,
  time_out,
)
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.mdp.actions import HeadPerturbActionCfg
from colosseum.mdp.velocity_command import CurriculumVelocityCommandCfg
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg

# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #

commands: Dict[str, CommandTermCfg] = {
  # booster_gym commands [vx, vy, ω_z] directly (no heading control). The grid
  # curriculum mirrors t1.py's _resample_curriculum_commands / _update_curriculum.
  "twist": CurriculumVelocityCommandCfg(
    entity_name="robot",
    heading_command=False,
    # booster has no heading/forward command styles — sample the grid directly
    # with only a standing fraction (still_proportion = 0.1).
    rel_standing_envs=0.1,
    rel_forward_envs=0.0,
    rel_heading_envs=0.0,
    debug_vis=True,
    resampling_time_range=(8.0, 12.0),
    ranges=CurriculumVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
    ),
    # booster T1.yaml grid params.
    curriculum=True,
    lin_levels=10,
    ang_levels=10,
    seed_lin_level=0,
    lin_vel_x_resolution=0.2,
    lin_vel_y_resolution=0.1,
    ang_vel_resolution=0.2,
    x_toler=0.4,
    y_toler=0.2,
    yaw_toler=0.2,
    episode_length_toler=0.1,
    update_rate=0.1,
    filter_weight=0.1,
  ),
  # Single gait clock (gait_frequency in [1, 2] Hz, zero for standing envs).
  "gait_phase": GaitPhaseCommandCfg(
    gait_freq_range=(1.0, 2.0),
    gate_command_name="twist",
    gate_speed_threshold=0.05,
  ),
}

# --------------------------------------------------------------------------- #
# Actions                                                                      #
# --------------------------------------------------------------------------- #

# booster_gym: target = default + action_scale * action, action_scale = 1.0.
# Applied to Waist + 12 leg joints (13 DOF). Arms are not selected, so they are
# held at their default pose by their PD actuators. The head is driven by a
# separate perturbation term (training robustness; held at default in play).
actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=("Waist", ".*Hip.*", ".*Knee.*", ".*Ankle.*"),
    scale=1.0,
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

# --------------------------------------------------------------------------- #
# Terminations                                                                 #
# --------------------------------------------------------------------------- #

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  # booster terminate_height = 0.45 (base below this => fall).
  "fell_over": TerminationTermCfg(
    func=root_height_below_minimum,
    params={"minimum_height": 0.45},
  ),
  "nan": TerminationTermCfg(func=nan_detection),
}
