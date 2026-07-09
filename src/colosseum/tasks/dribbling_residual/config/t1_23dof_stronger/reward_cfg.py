from mjlab.managers import RewardTermCfg

from colosseum.mdp.ball_rewards import (
  ball_kick_impulse,
  foot_ball_contact,
  robot_wrong_side_penalty,
)

from ..t1.reward_cfg import rewards as _base

rewards = dict(_base)

# Keep a *small* whole-foot contact bootstrap (below the inner-foot terms).
# Deleting it entirely (inner-only) starved discovery: foot5 is a narrow surface
# the policy can't stumble onto while the possession trackers are satisfiable by
# sole/toe nudges, so the inner rewards never got positive samples to climb on
# (they went flat). This dense, easy-to-find "touch the ball at all" gradient
# gets the ball to the feet; the higher-weighted inner terms then bias the touch
# onto the inside face. Weight kept low (0.3, was 0.5) so it scaffolds without
# out-voting the enforcement. Rebuilt as a fresh term (not a mutated weight) so
# we don't alias/modify the base task's shared RewardTermCfg object.
rewards["foot_ball_contact"] = RewardTermCfg(
  func=foot_ball_contact,
  weight=0.3,
  params={
    "sensor_name": "foot_ball_contact",
    "command_name": "ball_vel",
  },
)

# Bootstrap: pays for touching the ball *at all* on the inner face (foot5 medial
# capsule), giving a gradient before the ball is ever fast so the policy can
# climb from "graze the inside" toward "push the inside hard".
rewards["inner_foot_ball_contact"] = RewardTermCfg(
  func=foot_ball_contact,
  weight=1.0,
  params={
    "sensor_name": "foot_inner_ball_contact",
    "command_name": "ball_vel",
  },
)

# The one high-value contact reward: fast ball, in the commanded direction, ONLY
# credited while the touch is on the inner face (foot5). Latched credit window
# (borrowed from kicking_residual) pays for ~15 steps after each inner-foot
# strike scaled by the ball's target-directed speed, so the policy gets a dense
# "make each touch faster" gradient and re-arms it on every touch — riding along
# with continuous dribbling instead of replacing it. speed_ref=3.0 keeps the
# reward steep at achievable speeds (was 5.0, which flattened the incentive).
# This is what enforces inside-of-the-foot kicks: a sole/toe punt pays nothing.
rewards["ball_kick_impulse"] = RewardTermCfg(
  func=ball_kick_impulse,
  weight=2.0,
  params={
    "sensor_name": "foot_inner_ball_contact",
    "command_name": "ball_vel",
    "speed_ref": 3.0,
    "min_contact_force": 50.0,
    "credit_steps": 15,
  },
)

# Light penalty for being on the target side of the ball during the final
# approach, reinforcing the twist circumnavigation (approach from behind).
rewards["robot_wrong_side"] = RewardTermCfg(
  func=robot_wrong_side_penalty,
  weight=-1.5,
  params={"command_name": "ball_vel"},
)
