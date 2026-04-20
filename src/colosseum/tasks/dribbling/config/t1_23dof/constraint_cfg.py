"""CaT constraint terms for the Booster T1 dribbling task."""

from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.managers import ConstraintTermCfg
from colosseum.mdp import constraints
from colosseum.robots.t1_23dof.constants import FOOT_SITE_NAMES
from colosseum.tasks.dribbling.mdp.rewards import feet_distance_penalty

dribbling_constraints: dict[str, ConstraintTermCfg] = {
  # ------------------------------------------------------------------ #
  # Hard constraints (max_p = 1.0) — immediate termination on violation #
  # ------------------------------------------------------------------ #
  # Any self-collision between trunk subtrees.
  # "self_collision": ConstraintTermCfg(
  #   func=constraints.sensor_any_contact,
  #   max_p=1.0,
  #   params={"sensor_name": "self_collision", "force_threshold": 1.0},
  # ),
  # # Foot touching the other foot.
  # "foot_foot_contact": ConstraintTermCfg(
  #   func=constraints.sensor_any_contact,
  #   max_p=1.0,
  #   params={"sensor_name": "foot_foot_contact", "force_threshold": 1.0},
  # ),
  # # Non-foot body part touching the ball (wrong technique).
  # "nonfoot_ball_contact": ConstraintTermCfg(
  #   func=constraints.sensor_any_contact,
  #   max_p=1.0,
  #   params={"sensor_name": "nonfoot_ball_contact", "force_threshold": 1.0},
  # ),
  # # Non-foot body part touching the ground (robot has fallen).
  # "nonfoot_ground_contact": ConstraintTermCfg(
  #   func=constraints.sensor_any_contact,
  #   max_p=1.0,
  #   params={"sensor_name": "nonfoot_ground_touch", "force_threshold": 1.0},
  # ),
  # ------------------------------------------------------------------ #
  # Soft constraints — stochastic termination proportional to severity  #
  # ------------------------------------------------------------------ #
  # Base tilt: horizontal gravity component > 0.5 (≈ 30° from upright).
  # Adds pressure before the hard bad_orientation termination at 70°.
  # "base_orientation": ConstraintTermCfg(
  #   func=constraints.base_orientation,
  #   max_p=0.5,
  #   params={"limit": 0.5, "asset_cfg": SceneEntityCfg("robot")},
  # ),
  # Base height < 0.4 m — catches forward/sideways falls that stay
  # within the 70° orientation limit.
  # "min_base_height": ConstraintTermCfg(
  #   func=constraints.min_base_height,
  #   max_p=0.5,
  #   params={"limit": 0.45, "asset_cfg": SceneEntityCfg("robot")},
  # ),
  # Feet too close together (XY plane): replaces feet_distance reward.
  # "feet_distance": ConstraintTermCfg(
  #   func=feet_distance_penalty,
  #   max_p=0.25,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
  #     "min_dist": 0.05,
  #   },
  # ),
  # Joint range: hip and knee — 90% of physical range from default.
  # "joint_range_legs": ConstraintTermCfg(
  #   func=constraints.joint_range,
  #   max_p=0.25,
  #   params={
  #     "limit": 1.0,
  #     "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*Hip.*", r".*Knee.*")),
  #   },
  # ),
  # # Joint range: ankles — smaller physical range than hips/knees.
  # "joint_range_ankle": ConstraintTermCfg(
  #   func=constraints.joint_range,
  #   max_p=0.25,
  #   params={
  #     "limit": 0.8,
  #     "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*Ankle.*",)),
  #   },
  # ),
  # Joint range: upper body (shoulders, elbows, waist).
  # "joint_range_upper": ConstraintTermCfg(
  #   func=constraints.joint_range,
  #   max_p=0.25,
  #   params={
  #     "limit": 0.8,
  #     "asset_cfg": SceneEntityCfg(
  #       "robot",
  #       joint_names=(r".*Shoulder.*", r".*Elbow.*", r"Waist"),
  #     ),
  #   },
  # ),
  # Peak foot impact force > 600 N (≈ 1g for a 60 kg robot).
  # "foot_contact_force": ConstraintTermCfg(
  #   func=constraints.sensor_peak_force,
  #   max_p=0.25,
  #   params={"sensor_name": "feet_ground_contact", "limit": 600.0},
  # ),
}
