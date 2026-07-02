"""Rewards for the single-kick dribbling variant.

The goal is a single strike that sends the ball off at the *commanded* velocity
(fixed magnitude, slightly randomized — see cat_cfg) in the target direction, so
velocity tracking is the objective, not raw kick power. Reuses the base
t1-dribbling-residual rewards with:
  - ``penalty_hip_pose`` relaxed (-1.0 -> -0.5): the pose-deviation penalty
    otherwise fights the elongated strike stride the per-joint residual produces.
  - velocity-tracking terms bumped so matching the commanded ball velocity
    dominates: ``ball_vel_tracking`` (vector) and ``ball_vel_norm`` (magnitude).
  - ``ball_kicked_away`` success bonus: one-shot positive reward the step the
    ball-kicked-away termination fires.

The base ``ball_vel_*`` terms fire on the ball's post-kick flight (they gate on
ball speed, and the ball only moves once struck), so they already measure "did
the kick produce the commanded velocity vector" — no strike-gated term needed.
"""

from dataclasses import replace

from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.mdp.ball_rewards import (
  ball_kick_reach_penalty,
  stance_foot_ball_clearance_penalty,
)
from colosseum.tasks.dribbling_residual.mdp.terminations import ball_kicked_away_bonus

from ..t1_23dof.reward_cfg import rewards as _base_rewards

rewards = dict(_base_rewards)

# Relax the pose-deviation penalty so the swing leg can elongate the strike.
rewards["penalty_hip_pose"] = replace(rewards["penalty_hip_pose"], weight=-0.5)
rewards["feet_swing"] = replace(rewards["feet_swing"], weight=0.05)
rewards["feet_phase"] = replace(rewards["feet_phase"], weight=0.05)
rewards["arm_phase"] = replace(rewards["arm_phase"], weight=0.05)

# Make matching the commanded ball velocity the dominant objective.
rewards["ball_vel_tracking"] = replace(rewards["ball_vel_tracking"], weight=4.0)
rewards["ball_vel_norm"] = replace(rewards["ball_vel_norm"], weight=3.0)

# Elongate the step: penalize striking the ball when it is too close to the root
# along the kick direction, so the robot reaches out to meet it further ahead.
# min_reach is the elongation knob; keep the weight modest so the robot doesn't
# learn to avoid contact.
rewards["ball_kick_reach"] = RewardTermCfg(
  func=ball_kick_reach_penalty,
  weight=-5.0,
  params={
    "sensor_name": "foot_ball_contact",
    "command_name": "ball_vel",
    "min_reach": 0.4,
    "min_contact_force": 10.0,
    "credit_steps": 25,
  },
)

# Keep the stance foot clear of the ball: penalize both feet crowding it at
# once, so only the kicking foot engages. sigma is the clearance knob.
rewards["stance_foot_ball_clearance"] = RewardTermCfg(
  func=stance_foot_ball_clearance_penalty,
  weight=-2.0,
  params={
    "asset_cfg": SceneEntityCfg("robot", body_names=r"^(left|right)_foot_link$"),
    "sigma": 0.07,
  },
)

# One-shot success bonus when the struck ball leaves the radius.
rewards["ball_kicked_away"] = RewardTermCfg(
  func=ball_kicked_away_bonus,
  weight=10.0,
  params={"term_name": "ball_kicked_away"},
)
