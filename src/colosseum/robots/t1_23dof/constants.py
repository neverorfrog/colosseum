"""Booster T1 robot configuration.

Deploy only needs file paths and static constants. Training-only mjlab imports are
guarded so deploy can import this module without mjlab installed.
"""

import mujoco

try:  # pragma: no cover - train-only dependency
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
  from mjlab.utils.os import update_assets

  _MJLAB_AVAILABLE = True
except ImportError:
  _MJLAB_AVAILABLE = False

if _MJLAB_AVAILABLE:
  from colosseum.robots.t1_23dof.actuators import (
    T1_ACTUATOR_ANKLE_PITCH,
    T1_ACTUATOR_ANKLE_ROLL,
    T1_ACTUATOR_ARM,
    T1_ACTUATOR_HIP_PITCH,
    T1_ACTUATOR_HIP_ROLL,
    T1_ACTUATOR_HIP_YAW,
    T1_ACTUATOR_KNEE,
    T1_ACTUATOR_NECK,
    T1_ACTUATOR_WAIST,
  )
  from colosseum.robots.t1_23dof.collisions import FEET_ONLY_COLLISION, FEET_SELF_COLLISION
from colosseum.utils import src_dir

##
# MJCF and assets.
##

# Path to unified base XML (23-DOF full body)
# Locomotion (12-DOF) vs Full-body (23-DOF) is controlled by which actuators are added
XML = src_dir() / "robots" / "t1_23dof" / "xmls" / "T1_23dof.xml"

assert XML.exists(), f"XML not found: {XML}"


if _MJLAB_AVAILABLE:

  def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, XML.parent / "assets", meshdir)
    return assets


if _MJLAB_AVAILABLE:

  def get_spec() -> mujoco.MjSpec:
    """Load T1 base model (23 DOF structure, actuators added via Python)."""
    spec = mujoco.MjSpec.from_file(str(XML))
    # Ensure no XML actuators are present (they should be commented in XML)
    spec.actuators.clear()
    return spec


##
# Actuator config.
##

# Booster T1 23-DOF joint names (extracted from XML)
JOINT_NAMES = [
  # Lower body (legs) - 10 DOF
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
  # Torso - 1 DOF
  "Waist",
  # Upper body (arms) - 8 DOF
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  # Head - 2 DOF
  "AAHead_yaw",
  "Head_pitch",
]


if _MJLAB_AVAILABLE:
  # 23-DOF Full Body
  ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
      T1_ACTUATOR_NECK,
      T1_ACTUATOR_ARM,
      T1_ACTUATOR_WAIST,
      T1_ACTUATOR_HIP_PITCH,
      T1_ACTUATOR_HIP_ROLL,
      T1_ACTUATOR_HIP_YAW,
      T1_ACTUATOR_KNEE,
      T1_ACTUATOR_ANKLE_PITCH,
      T1_ACTUATOR_ANKLE_ROLL,
    ),
    soft_joint_pos_limit_factor=0.9,
  )

##
# Keyframe config
##

# Home keyframe joint positions (from XML keyframe)
HOME_QPOS: dict[str, float] = {
  # Head
  "AAHead_yaw": 0.0,
  "Head_pitch": 0.0,
  # Left arm (manufacturer's deployment values)
  "Left_Shoulder_Pitch": 0.2,
  "Left_Shoulder_Roll": -1.3,
  "Left_Elbow_Pitch": 0.0,
  "Left_Elbow_Yaw": -0.5,
  # Right arm (manufacturer's deployment values)
  "Right_Shoulder_Pitch": 0.2,
  "Right_Shoulder_Roll": 1.3,
  "Right_Elbow_Pitch": 0.0,
  "Right_Elbow_Yaw": 0.5,
  # Waist
  "Waist": 0.0,
  # Left leg
  "Left_Hip_Pitch": -0.2,
  "Left_Hip_Roll": 0.0,
  "Left_Hip_Yaw": 0.0,
  "Left_Knee_Pitch": 0.4,
  "Left_Ankle_Pitch": -0.2,
  "Left_Ankle_Roll": 0.0,
  # Right leg
  "Right_Hip_Pitch": -0.2,
  "Right_Hip_Roll": 0.0,
  "Right_Hip_Yaw": 0.0,
  "Right_Knee_Pitch": 0.4,
  "Right_Ankle_Pitch": -0.2,
  "Right_Ankle_Roll": 0.0,
}

##
# Robot Configuration Functions
##

if _MJLAB_AVAILABLE:

  def get_robot_cfg(foot_self_collision: bool = False) -> EntityCfg:
    collision = FEET_SELF_COLLISION if foot_self_collision else FEET_ONLY_COLLISION
    return EntityCfg(
      init_state=EntityCfg.InitialStateCfg(
        pos=(0, 0, 0.665),
        joint_pos=HOME_QPOS,
        joint_vel={".*": 0.0},
      ),
      collisions=(collision,),
      spec_fn=get_spec,
      articulation=ARTICULATION,
    )

  # Convenience shorthand
  ROBOT_CFG = get_robot_cfg()

##
# Action Scale (uniform for all joints, matching mjlab)
##

ACTION_SCALE: dict[str, float] = {name: 0.25 for name in JOINT_NAMES}

##
# Foot geom names (for events like friction randomization)
##

# All foot geometry names including sphere contacts
# Each foot has 1 main link + 5 contact spheres for stable multi-point contact
FOOT_GEOM_NAMES = (
  "left_foot_link",
  "left_foot_sphere_1_link",
  "left_foot_sphere_2_link",
  "left_foot_sphere_3_link",
  "left_foot_sphere_4_link",
  "left_foot_sphere_5_link",
  "right_foot_link",
  "right_foot_sphere_1_link",
  "right_foot_sphere_2_link",
  "right_foot_sphere_3_link",
  "right_foot_sphere_4_link",
  "right_foot_sphere_5_link",
)

FOOT_SITE_NAMES = ("left_foot", "right_foot")

BASE_BODY_NAME = "Trunk"
