import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, nan_detection, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.mdp.velocity_command import CurriculumVelocityCommandCfg
from colosseum.robots.t1_23dof.constants import JOINT_NAMES
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
    speed_range=(0.1, 1.0),
  ),
  "gait_phase": GaitPhaseCommandCfg(
    gait_freq_range=(1.5, 2.0),
    gate_command_name="twist",
    gate_speed_threshold=0.05,
  ),
}

FIXED_JOINTS = {
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
}
UNACTUATED_JOINTS = {"AAHead_yaw", "Head_pitch"}
ACTION_SCALE: dict[str, float] = {
  name: (0.0 if name in FIXED_JOINTS else 0.25) for name in JOINT_NAMES if name not in UNACTUATED_JOINTS
}


actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=("^(?!AAHead_yaw$|Head_pitch$).*$",),
    scale=ACTION_SCALE,
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
