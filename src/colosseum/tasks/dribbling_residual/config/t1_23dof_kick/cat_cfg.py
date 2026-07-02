"""Commands / actions / terminations for the single-kick dribbling variant.

Reuses the base t1-dribbling-residual commands and actions, and adds a
``ball_kicked_away`` success termination: once the robot has struck the ball, the
episode ends when the ball leaves a radius around the robot (it got kicked away).
"""

from mjlab.managers.termination_manager import TerminationTermCfg

from colosseum.tasks.dribbling_residual.mdp.ball_velocity_command import (
  BallVelocityCommandCfg,
)
from colosseum.tasks.dribbling_residual.mdp.terminations import BallKickedAway

from ..t1_23dof.cat_cfg import actions
from ..t1_23dof.cat_cfg import commands as _base_commands
from ..t1_23dof.cat_cfg import terminations as _base_terminations

# Fixed-velocity kick: one constant speed per episode (slightly randomized) in
# the target direction, so following the commanded velocity means "kick the ball
# to this steady velocity" rather than dribbling it to a spot.
commands = dict(_base_commands)
commands["ball_vel"] = BallVelocityCommandCfg(
  constant_speed=True,
  speed_range=(2.0, 5.0),
)

terminations = dict(_base_terminations)
terminations["ball_kicked_away"] = TerminationTermCfg(
  func=BallKickedAway,
  params={"success_radius": 1.5, "sensor_name": "foot_ball_contact"},
)

__all__ = ["actions", "commands", "terminations"]
