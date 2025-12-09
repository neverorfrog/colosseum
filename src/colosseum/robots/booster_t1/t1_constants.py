"""Booster T1 robot configuration for mjlab."""

import mujoco
import mujoco.viewer as viewer
from pathlib import Path

from mjlab.entity import EntityCfg, Entity, EntityArticulationInfoCfg
from mjlab.actuator import XmlPositionActuatorCfg

from colosseum.utils import src_dir
from colosseum.robots.booster_t1.t1_contacts import (
    FEET_ONLY_COLLISION,
    FULL_COLLISION,
    FULL_COLLISION_WITHOUT_SELF,
)

# Path to unified base XML (23-DOF full body)
# Locomotion (12-DOF) vs Full-body (23-DOF) is controlled by which actuators are added
T1_BASE_XML = src_dir() / "robots" / "booster_t1" / "xmls" / "T1_23dof.xml"

assert T1_BASE_XML.exists(), f"XML not found: {T1_BASE_XML}"

##
# Spec functions
##

def get_t1_spec() -> mujoco.MjSpec:
    """Load T1 base model (23 DOF structure, actuators added via Python)."""
    return mujoco.MjSpec.from_file(str(T1_BASE_XML))

##
# Keyframe config
##

LOCOMOTION_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.665),
    joint_pos={
        ".*Hip_Pitch": -0.2,
        ".*Knee_Pitch": 0.4,
        ".*Ankle_Pitch": -0.2,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULLBODY_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.665),
    joint_pos={
        # Arms (explicit names to ensure they're set correctly)
        "Left_Shoulder_Roll": -0.4,
        "Left_Elbow_Yaw": -0.4,
        "Right_Shoulder_Roll": 0.4,
        "Right_Elbow_Yaw": 0.4,
        # Legs (regex patterns)
        ".*Hip_Pitch": -0.2,
        ".*Knee_Pitch": 0.4,
        ".*Ankle_Pitch": -0.2,
    },
    joint_vel={".*": 0.0},
)


##
# Articulation Configurations
##

# 23-DOF Full Body - uses actuators defined in XML
# The XML contains position actuators with kp=75, kv=5 for all 23 joints
T1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        XmlPositionActuatorCfg(
            joint_names_expr=(
                ".*AAHead_yaw",
                ".*Head_pitch",
                ".*Left_Shoulder_Pitch",
                ".*Left_Shoulder_Roll",
                ".*Left_Elbow_Pitch",
                ".*Left_Elbow_Yaw",
                ".*Right_Shoulder_Pitch",
                ".*Right_Shoulder_Roll",
                ".*Right_Elbow_Pitch",
                ".*Right_Elbow_Yaw",
                ".*Waist",
                ".*Left_Hip_Pitch",
                ".*Left_Hip_Roll",
                ".*Left_Hip_Yaw",
                ".*Left_Knee_Pitch",
                ".*Left_Ankle_Pitch",
                ".*Left_Ankle_Roll",
                ".*Right_Hip_Pitch",
                ".*Right_Hip_Roll",
                ".*Right_Hip_Yaw",
                ".*Right_Knee_Pitch",
                ".*Right_Ankle_Pitch",
                ".*Right_Ankle_Roll",
            ),
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)

##
# Robot Configuration Functions
##

def get_t1_robot_cfg() -> EntityCfg:
    """
    Get T1 robot config (23 DOF full body).

    Uses actuators defined in the XML with kp=75, kv=5 for all joints.
    This matches the mjlab approach for velocity tracking tasks.
    """
    return EntityCfg(
        init_state=FULLBODY_HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_t1_spec,
        articulation=T1_ARTICULATION,
    )

# Convenience shorthand
T1_ROBOT_CFG = get_t1_robot_cfg()

##
# Action Scale (uniform for all joints, matching mjlab)
##

# All 23 joints use the same action scale of 0.25
T1_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
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

T1_ACTION_SCALE: dict[str, float] = {name: 0.25 for name in T1_JOINT_NAMES}

if __name__ == "__main__":
    robot = Entity(T1_ROBOT_CFG)
    model = robot.spec.compile()
    viewer.launch(model)
