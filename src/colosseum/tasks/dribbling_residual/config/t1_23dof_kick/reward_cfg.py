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
  ball_kick_impulse,
  ball_kick_reach_penalty,
  foot_ball_contact,
  stance_foot_ball_clearance_penalty,
)
from colosseum.mdp.rewards import pose_deviation_penalty
from colosseum.tasks.dribbling_residual.mdp.terminations import ball_kicked_away_bonus

from ..t1_23dof.reward_cfg import rewards as _base_rewards

rewards = dict(_base_rewards)

# Relax the pose-deviation penalty so the swing leg can elongate the strike —
# but only for hip yaw/roll (the stride joints). Pitch/knee stay at the base
# task's effective strength (0.5 * 1.0) to cap leg elongation, and ankle roll
# at full strength (0.5 * 4.0 = base 2.0) against erratic ankle flicks.
# Fresh term (not replace) so we don't mutate the base task's shared params.
rewards["penalty_hip_pose"] = RewardTermCfg(
  func=pose_deviation_penalty,
  weight=-0.5,
  params={
    "asset_cfg": SceneEntityCfg(
      "robot",
      joint_names=(
        ".*Hip_Roll",
        ".*Hip_Yaw",
        ".*Hip_Pitch",
        ".*Knee_Pitch",
        ".*Ankle_Roll",
      ),
    ),
    "weights_standing": {
      ".*Hip_Yaw": 5.0,
      ".*Hip_Roll": 2.0,
      ".*Hip_Pitch": 1.0,
      ".*Knee_Pitch": 1.0,
      ".*Ankle_Roll": 4.0,
    },
  },
)
rewards["feet_swing"] = replace(rewards["feet_swing"], weight=0.05)
rewards["feet_phase"] = replace(rewards["feet_phase"], weight=0.05)
rewards["arm_phase"] = replace(rewards["arm_phase"], weight=0.05)

# Make matching the commanded ball velocity the dominant objective.
rewards["ball_vel_tracking"] = replace(rewards["ball_vel_tracking"], weight=4.0)
rewards["ball_vel_norm"] = replace(rewards["ball_vel_norm"], weight=3.0)

# Inside-foot kick enforcement (recipe from t1_23dof_stronger): shrink the
# whole-foot contact bootstrap so it scaffolds discovery without out-voting the
# inner-face terms, add an inner-face (foot5 medial capsule) contact bootstrap,
# and pay the high-value strike credit only for inner-face contact — a sole/toe
# punt earns no impulse reward. Fresh terms, not mutated base cfgs.
rewards["foot_ball_contact"] = RewardTermCfg(
  func=foot_ball_contact,
  weight=0.3,
  params={
    "sensor_name": "foot_ball_contact",
    "command_name": "ball_vel",
  },
)
rewards["inner_foot_ball_contact"] = RewardTermCfg(
  func=foot_ball_contact,
  weight=1.0,
  params={
    "sensor_name": "foot_inner_ball_contact",
    "command_name": "ball_vel",
  },
)
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
    "min_reach": 0.3,  # 0.4 over-elongated the leg on the real robot.
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
