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
  foot_ball_contact,
  stance_foot_placement_penalty,
)
from colosseum.mdp.rewards import pose_deviation_penalty
from colosseum.tasks.dribbling_residual.mdp.rewards import (
  feet_phase_ball_gated,
  feet_swing_ball_gated,
)
from colosseum.tasks.dribbling_residual.mdp.terminations import ball_kicked_away_bonus

from ..t1_23dof.reward_cfg import rewards as _base_rewards

rewards = dict(_base_rewards)

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
      ".*Hip_Roll": 1.0,
      ".*Hip_Pitch": 1.0,
      ".*Knee_Pitch": 1.0,
      ".*Ankle_Roll": 4.0,
    },
  },
)
rewards["feet_swing"] = replace(
  rewards["feet_swing"],
  func=feet_swing_ball_gated,
  weight=1.0,
  params={**rewards["feet_swing"].params, "gate_distance": 1.0},
)
rewards["feet_phase"] = replace(
  rewards["feet_phase"],
  func=feet_phase_ball_gated,
  weight=1.0,
  params={**rewards["feet_phase"].params, "gate_distance": 1.0},
)
rewards["arm_phase"] = replace(rewards["arm_phase"], weight=0.5)

# Make matching the commanded ball velocity the dominant objective.
rewards["ball_vel_tracking"] = replace(rewards["ball_vel_tracking"], weight=4.0)
rewards["ball_vel_norm"] = replace(rewards["ball_vel_norm"], weight=3.0)

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

# Stance-foot placement: the support (grounded) foot should sit on the line
# through the ball center orthogonal to the commanded kick direction, at least
# lateral_margin to the side (side and width are the policy's choice). Replaces
# ball_kick_reach + stance_foot_ball_clearance with one setup-geometry term.
rewards["stance_foot_placement"] = RewardTermCfg(
  func=stance_foot_placement_penalty,
  weight=-3.0,
  params={
    "command_name": "ball_vel",
    "asset_cfg": SceneEntityCfg("robot", body_names=r"^(left|right)_foot_link$"),
    "sensor_name": "feet_ground_contact",
    "engage_distance": 0.6,
    "lateral_margin": 0.15,
  },
)

# One-shot success bonus when the struck ball leaves the radius.
rewards["ball_kicked_away"] = RewardTermCfg(
  func=ball_kicked_away_bonus,
  weight=10.0,
  params={"term_name": "ball_kicked_away"},
)
