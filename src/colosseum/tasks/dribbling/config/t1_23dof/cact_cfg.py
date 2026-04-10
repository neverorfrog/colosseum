import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.robots.t1_23dof.constants import ACTION_SCALE
from colosseum.tasks.dribbling.mdp.ball_velocity_command import BallVelocityCommandCfg
from colosseum.tasks.dribbling.mdp.gait_phase_command import GaitPhaseCommandCfg

_ARM_JOINTS = {
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
}
# Head joints need larger scale so the policy can pitch down to look at the ball at foot level.
# Head_pitch range is [-0.35, 1.22] rad; scale=1.2 covers the full downward range.
# AAHead_yaw range is [-1.57, 1.57] rad; scale=1.0 covers ±57° lateral tracking.
_HEAD_SCALE = {"AAHead_yaw": 1.0, "Head_pitch": 1.2}
# Arms scale=0: with use_default_offset=True, target = HOME_QPOS + 0*action = HOME_QPOS always.
_DRIBBLING_ACTION_SCALE = {
  k: (0.0 if k in _ARM_JOINTS else _HEAD_SCALE.get(k, v)) for k, v in ACTION_SCALE.items()
}

commands: Dict[str, CommandTermCfg] = {
  "ball_vel": BallVelocityCommandCfg(
    robot_entity="robot",
    ball_entity="ball",
    speed_range=(0.3, 1.0),
    heading_range=math.pi / 3,
    resampling_time_range=(10.0, 20.0),
    debug_vis=True,
  ),
  "gait_phase": GaitPhaseCommandCfg(),
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=_DRIBBLING_ACTION_SCALE,
    use_default_offset=True,
  )
}

curriculum = {}

terminations = {
  "time_out": TerminationTermCfg(func=time_out, time_out=True),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  ),
}
