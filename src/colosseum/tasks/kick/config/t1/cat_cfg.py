"""Commands / actions / terminations for the single-kick dribbling variant.

Reuses the base t1-dribbling-residual commands and actions, and adds a
``ball_kicked_away`` success termination: once the robot has struck the ball, the
episode ends when the ball leaves a radius around the robot (it got kicked away).
"""

from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.tasks.kick.mdp.ball_velocity_command import (
  BallVelocityCommandCfg,
)
from colosseum.tasks.kick.mdp.terminations import BallKickedAway


import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, nan_detection, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.robots.t1.constants import ACTION_SCALE
from colosseum.tasks.locomotion.mdp.gait_phase_command import GaitPhaseCommandCfg
from colosseum.tasks.kick.mdp.ball_twist_command import (
  BallTwistCommandCfg,
)
from colosseum.tasks.kick.mdp.ball_velocity_command import (
  BallVelocityCommandCfg,
)
from colosseum.tasks.kick.mdp.head_ik_action import HeadIKActionCfg
from colosseum.tasks.kick.mdp.terminations import BallLostTermination

commands: Dict[str, CommandTermCfg] = {
  "twist": BallTwistCommandCfg(stop_distance=0.25, approach_offset=0.4),
  "ball_vel": BallVelocityCommandCfg(
    speed_range=(2.0, 5.0),
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
  "head_ik": HeadIKActionCfg(),
}

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
  "nan": TerminationTermCfg(func=nan_detection),
  "ball_lost": TerminationTermCfg(
    func=BallLostTermination,
    params={"loss_timeout": 3.0},
  ),
  "ball_kicked_away": TerminationTermCfg(
    func=BallKickedAway,
    params={"credit_steps": 50, "settle_steps": 75, "sensor_name": "foot_ball_contact"},
  )
}

