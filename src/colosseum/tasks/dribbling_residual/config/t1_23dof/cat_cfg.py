import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, nan_detection, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from colosseum.mdp.velocity_command import CurriculumVelocityCommandCfg
from colosseum.robots.t1_23dof.constants import ACTION_SCALE
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg
from colosseum.tasks.dribbling_residual.mdp.ball_twist_command import (
  BallTwistCommandCfg,
)
from colosseum.tasks.dribbling_residual.mdp.ball_velocity_command import (
  BallVelocityCommandCfg,
)
from colosseum.tasks.dribbling_residual.mdp.head_ik_action import HeadIKActionCfg

commands: Dict[str, CommandTermCfg] = {
  "twist": BallTwistCommandCfg(stop_distance=0.25),
  "ball_vel": BallVelocityCommandCfg(
    resampling_time_range=(5.0, 10.0),
    target_reached_threshold=0.5,
    speed_range=(0.1, 0.5),
  ),
  "gait_phase": GaitPhaseCommandCfg(
    gait_freq_range=(1.5, 2.0),
    gate_command_name="twist",
    gate_speed_threshold=0.05,
  ),
}

# Head joints get scale=0: their targets are overridden by HeadIKActionCfg, so
# the policy (frozen walk + residual) should not waste action on them.
_HEAD_JOINTS = {"AAHead_yaw", "Head_pitch"}
VELOCITY_ACTION_SCALE = {
  k: (0.0 if k in _HEAD_JOINTS else v) for k, v in ACTION_SCALE.items()
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=VELOCITY_ACTION_SCALE,
    use_default_offset=True,
  ),
  # IK head tracking: runs after joint_pos and overrides the head targets with
  # analytic yaw/pitch toward the ball, returning to (0, 0) when out of view.
  "head_ik": HeadIKActionCfg(),
}

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
  "nan": TerminationTermCfg(func=nan_detection),
}
