"""Events for the single-kick dribbling variant.

Reuses the base t1-dribbling-residual events but spawns the ball near the robot
(within ~2 m) so each episode is a short approach + one kick, not a long walk-in.

The robot resets in a +-0.5 m box (``reset_base``); a +-0.9 m ball box keeps the
worst-case robot->ball separation at sqrt(2) * (0.9 + 0.5) ~= 1.98 m <= 2 m
(env_origins cancel, so only the two sampled offsets matter).
"""

from dataclasses import replace

from ..t1_23dof.event_cfg import events as _base_events

events = dict(_base_events)

_reset_ball = events["reset_ball"]
events["reset_ball"] = replace(
  _reset_ball,
  params={
    **_reset_ball.params,
    "pose_range": {"x": (0.5, 1.5), "y": (-1.0, 1.0), "z": (0.11, 0.11)},
  },
)
