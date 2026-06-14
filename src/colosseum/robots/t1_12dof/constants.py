"""Booster T1 12-DOF (legs-only) robot configuration.

Mirrors booster_gym's ``T1_locomotion`` model: 12 leg joints, arms/head/waist
welded into the trunk (visual only). Used by the t1-velocity-12dof task.
"""

import mujoco

try:  # pragma: no cover - train-only dependency
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  _MJLAB_AVAILABLE = True
except ImportError:
  _MJLAB_AVAILABLE = False

if _MJLAB_AVAILABLE:
  from colosseum.robots.t1_12dof.actuators import LOCOMOTION_ACTUATORS
  from colosseum.robots.t1_12dof.collisions import (
    FEET_ONLY_COLLISION,
    FULL_COLLISION_WITHOUT_SELF,
  )
from colosseum.utils import src_dir

##
# MJCF and assets.
##

XML = src_dir() / "robots" / "t1_12dof" / "xmls" / "T1_12dof.xml"
assert XML.exists(), f"XML not found: {XML}"


if _MJLAB_AVAILABLE:

  def get_spec() -> mujoco.MjSpec:
    """Load the 12-DOF locomotion model (actuators added via Python)."""
    spec = mujoco.MjSpec.from_file(str(XML))
    spec.actuators.clear()
    return spec


##
# Joints.
##

# 12 leg joints in MuJoCo XML depth-first order.
JOINT_NAMES = [
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_Pitch",
  "Left_Ankle_Roll",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_Pitch",
  "Right_Ankle_Roll",
]

##
# Articulation.
##

if _MJLAB_AVAILABLE:
  # Hand-tuned gains matching booster_gym (hip 200/5, knee 200/5, ankle 50/1).
  LOCOMOTION_ARTICULATION = EntityArticulationInfoCfg(
    actuators=LOCOMOTION_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
  )

##
# Keyframe / default pose (booster_gym convention).
##

# Hip_Pitch=-0.2, Knee=0.4, Ankle_Pitch=-0.25; everything else 0.
HOME_QPOS: dict[str, float] = {
  "Left_Hip_Pitch": -0.2,
  "Left_Hip_Roll": 0.0,
  "Left_Hip_Yaw": 0.0,
  "Left_Knee_Pitch": 0.4,
  "Left_Ankle_Pitch": -0.25,
  "Left_Ankle_Roll": 0.0,
  "Right_Hip_Pitch": -0.2,
  "Right_Hip_Roll": 0.0,
  "Right_Hip_Yaw": 0.0,
  "Right_Knee_Pitch": 0.4,
  "Right_Ankle_Pitch": -0.25,
  "Right_Ankle_Roll": 0.0,
}

# booster_gym init pos z (m).
BASE_HEIGHT = (0.0, 0.0, 0.72)

##
# Robot configuration.
##

if _MJLAB_AVAILABLE:

  def get_robot_cfg(full_collision: bool = False) -> EntityCfg:
    collision = (
      FULL_COLLISION_WITHOUT_SELF if full_collision else FEET_ONLY_COLLISION
    )
    return EntityCfg(
      init_state=EntityCfg.InitialStateCfg(
        pos=BASE_HEIGHT,
        joint_pos=HOME_QPOS,
        joint_vel={".*": 0.0},
      ),
      collisions=(collision,),
      spec_fn=get_spec,
      articulation=LOCOMOTION_ARTICULATION,
    )

##
# Action scale: target = scale * action + default.
##

# booster_gym uses a single action_scale of 1.0 on all leg joints.
LOCOMOTION_ACTION_SCALE: dict[str, float] = {name: 1.0 for name in JOINT_NAMES}

##
# Symmetry (left-right mirror about the sagittal plane).
##

SYMMETRY_JOINT_NAMES: dict[str, str] = {
  "Left_Hip_Pitch": "Right_Hip_Pitch",
  "Left_Hip_Roll": "Right_Hip_Roll",
  "Left_Hip_Yaw": "Right_Hip_Yaw",
  "Left_Knee_Pitch": "Right_Knee_Pitch",
  "Left_Ankle_Pitch": "Right_Ankle_Pitch",
  "Left_Ankle_Roll": "Right_Ankle_Roll",
  "Right_Hip_Pitch": "Left_Hip_Pitch",
  "Right_Hip_Roll": "Left_Hip_Roll",
  "Right_Hip_Yaw": "Left_Hip_Yaw",
  "Right_Knee_Pitch": "Left_Knee_Pitch",
  "Right_Ankle_Pitch": "Left_Ankle_Pitch",
  "Right_Ankle_Roll": "Left_Ankle_Roll",
}

# Roll/yaw axes are pseudovectors: their sign flips under left-right mirroring.
FLIP_SIGN_JOINT_NAMES: list[str] = [
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Left_Ankle_Roll",
  "Right_Ankle_Roll",
]

##
# Foot / body names.
##

FOOT_GEOM_NAMES = (
  "left_foot_link",
  "left_foot1_collision",
  "left_foot2_collision",
  "left_foot3_collision",
  "left_foot4_collision",
  "left_foot5_collision",
  "right_foot_link",
  "right_foot1_collision",
  "right_foot2_collision",
  "right_foot3_collision",
  "right_foot4_collision",
  "right_foot5_collision",
)

FOOT_SITE_NAMES = ("left_foot", "right_foot")
FOOT_BODY_NAMES = ("left_foot_link", "right_foot_link")
BASE_BODY_NAME = "Trunk"
