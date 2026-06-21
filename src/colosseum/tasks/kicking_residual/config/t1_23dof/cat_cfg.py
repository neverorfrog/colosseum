import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, nan_detection, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.mdp.velocity_command import CurriculumVelocityCommandCfg
from colosseum.robots.t1_23dof.constants import MANUFACTURER_ACTION_SCALE
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg
from colosseum.tasks.kicking_residual.mdp.ball_twist_command import (
  BallTwistCommandCfg,
)
from colosseum.tasks.kicking_residual.mdp.ball_twist_command_forward import (
  BallForwardTwistCommandCfg,
)
from colosseum.tasks.kicking_residual.mdp.ball_velocity_command import (
  BallVelocityCommandCfg,
)
from colosseum.tasks.kicking_residual.mdp.commands import BallAngleCommandCfg
from colosseum.tasks.kicking_residual.mdp.head_ik_action import HeadIKActionCfg
from colosseum.tasks.kicking_residual.mdp.terminations import terminate_after_kick

# twist steers the frozen walk straight at the ball (same proven command as the
# dribbling residual); the kick *direction* is handled by the rewards + residual,
# not by engineering the approach. ball_angle stays for the obs/reward goal dir.
commands: Dict[str, CommandTermCfg] = {
  "ball_angle": BallAngleCommandCfg(debug_vis=True),
  "twist": BallTwistCommandCfg(stop_distance=0.25),
  "gait_phase": GaitPhaseCommandCfg(
    gait_freq_range=(1.5, 2.0),
    gate_command_name="twist",
    gate_speed_threshold=0.05,
  ),
}

# Head joints are excluded from the policy's action space entirely: HeadIKActionCfg
# drives them directly (action_dim=0), matching the velocity task's action layout
# (and the walk checkpoint's action head, which has no head outputs).
ARM_JOINT_NAMES = (
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
)
UNACTUATED_JOINTS = ("AAHead_yaw", "Head_pitch")
# Per-joint manufacturer action scales, with head joints dropped (driven by
# HeadIKActionCfg) and arms fixed at default pose (scale 0 -> targets = default,
# inert), matching the walk checkpoint's action head: those output dims got zero
# gradient during its training, so their values are untrained noise that must not
# move the arms.
VELOCITY_ACTION_SCALE = {
  name: (0.0 if name in ARM_JOINT_NAMES else scale)
  for name, scale in MANUFACTURER_ACTION_SCALE.items()
  if name not in UNACTUATED_JOINTS
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=("^(?!AAHead_yaw$|Head_pitch$).*$",),
    scale=VELOCITY_ACTION_SCALE,
    use_default_offset=True,
  ),
  # IK head tracking: drives the head joints directly with analytic yaw/pitch
  # toward the ball, returning to (0, 0) when out of view.
  "head_ik": HeadIKActionCfg(),
}

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
  "nan": TerminationTermCfg(func=nan_detection),
  # One strike per episode: terminate 2 s (100 control steps @ 50 Hz) after the
  # first foot-ball contact. Kills repeated-kick farming (the crouch-and-hug
  # pathology). min_contact_force must match ball_speed_kick's gate, and
  # delay_steps must exceed its kick_credit_steps so the credit window resolves.
  "kicked": TerminationTermCfg(
    func=terminate_after_kick,
    params={
      "sensor_name": "foot_ball_contact",
      "min_contact_force": 250.0,
      "delay_steps": 60,
    },
  ),
}
