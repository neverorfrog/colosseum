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
from colosseum.tasks.dribbling.mdp.head_ik_action import HeadIKActionCfg
from colosseum.tasks.dribbling.mdp.terminations import (
  ball_target_reached,
)

HEAD_JOINTS = {"AAHead_yaw", "Head_pitch"}
# head scale=0: head targets are overridden by HeadIKActionCfg below.
DRIBBLING_ACTION_SCALE = {
  k: (0.0 if k in HEAD_JOINTS else v) for k, v in ACTION_SCALE.items()
}

commands: Dict[str, CommandTermCfg] = {
  "ball_vel": BallVelocityCommandCfg(
    robot_entity="robot",
    ball_entity="ball",
    speed_range=(0.2, 1.0),
    target_distance_range=(2.5, 5.0),
    speed_gain=1.0,
    target_reached_threshold=0.5,
    heading_range=math.pi / 2,  # ± range around the robot forward direction
    resampling_time_range=(10.0, 10.0),
    debug_vis=True,
  ),
  "gait_phase": GaitPhaseCommandCfg(gait_freq_range=(1.5, 2.0)),
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=DRIBBLING_ACTION_SCALE,
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
  "ball_target_reached": TerminationTermCfg(
    func=ball_target_reached,
    params={"command_name": "ball_vel"},
  ),
}
