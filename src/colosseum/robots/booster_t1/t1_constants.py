"""Booster T1 robot configuration for mjlab."""

import mujoco
import mujoco.viewer as viewer
from pathlib import Path
from mjlab.utils.os import update_assets

from mjlab.entity import EntityCfg, Entity, EntityArticulationInfoCfg
from mjlab.actuator import XmlPositionActuatorCfg

from colosseum.utils import src_dir
from colosseum.robots.booster_t1.t1_contacts import (
    FEET_ONLY_COLLISION,
)

from colosseum.robots.booster_t1.t1_actuators import (
    T1_ACTUATOR_HIP_PITCH,
    T1_ACTUATOR_HIP_ROLL,
    T1_ACTUATOR_HIP_YAW,
    T1_ACTUATOR_KNEE,
    T1_ACTUATOR_ANKLE_PITCH,
    T1_ACTUATOR_ANKLE_ROLL,
    T1_ACTUATOR_NECK,
    T1_ACTUATOR_ARM,
    T1_ACTUATOR_WAIST,
)

##
# MJCF and assets.
##

# Path to unified base XML (23-DOF full body)
# Locomotion (12-DOF) vs Full-body (23-DOF) is controlled by which actuators are added
XML = src_dir() / "robots" / "booster_t1" / "xmls" / "T1_23dof.xml"

assert XML.exists(), f"XML not found: {XML}"

def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, XML.parent / "assets", meshdir)
  return assets

def get_spec() -> mujoco.MjSpec:
    """Load T1 base model (23 DOF structure, actuators added via Python)."""
    return mujoco.MjSpec.from_file(str(XML))

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
    # Left arm
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.4,
    # Right arm
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.4,
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

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.665),
    joint_pos=HOME_QPOS,
    joint_vel={".*": 0.0},
)


##
# Robot Configuration Functions
##

def get_robot_cfg() -> EntityCfg:
    """
    Get T1 robot config (23 DOF full body).

    Uses actuators defined in the XML with kp=75, kv=5 for all joints.
    This matches the mjlab approach for velocity tracking tasks.
    """
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
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


if __name__ == "__main__":
    robot = Entity(ROBOT_CFG)
    model = robot.spec.compile()
    viewer.launch(model)
