"""Booster T1 robot configuration for mjlab."""

import mujoco
import mujoco.viewer as viewer
from pathlib import Path

from mjlab.entity import EntityCfg, Entity, EntityArticulationInfoCfg
from mjlab.actuator import BuiltinPositionActuatorCfg

from colosseum.utils import src_dir
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
        "Left_Shoulder_Roll": -1.0,
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

# 12-DOF Locomotion (legs only)
T1_LOCOMOTION_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        T1_ACTUATOR_HIP_PITCH,
        T1_ACTUATOR_HIP_ROLL,
        T1_ACTUATOR_HIP_YAW,
        T1_ACTUATOR_KNEE,
        T1_ACTUATOR_ANKLE_PITCH,
        T1_ACTUATOR_ANKLE_ROLL,
    ),
    soft_joint_pos_limit_factor=0.9,
)

# 23-DOF Full Body (head + arms + waist + legs)
T1_FULLBODY_ARTICULATION = EntityArticulationInfoCfg(
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
# Robot Configuration Functions
##

def get_t1_locomotion_robot_cfg() -> EntityCfg:
    """
    Get T1 locomotion config (12 DOF legs only) - for training.

    Uses the full 23-DOF XML base but only adds actuators for the 12 leg joints.
    Upper body joints (head, arms, waist) remain passive.
    """
    return EntityCfg(
        init_state=LOCOMOTION_HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_t1_spec,
        articulation=T1_LOCOMOTION_ARTICULATION,
    )

def get_t1_fullbody_robot_cfg() -> EntityCfg:
    """
    Get T1 full body config (23 DOF) - for deployment/manipulation.

    Uses the full 23-DOF XML base and adds actuators for all joints
    (head, arms, waist, and legs).
    """
    return EntityCfg(
        init_state=FULLBODY_HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_t1_spec,
        articulation=T1_FULLBODY_ARTICULATION,
    )

# Convenience shorthands
T1_ROBOT_CFG = get_t1_fullbody_robot_cfg()
# T1_FULLBODY_CFG = get_t1_fullbody_robot_cfg()

# Compute ACTION_SCALE dictionary from actuator configs (for locomotion)
T1_ACTION_SCALE: dict[str, float] = {}
for a in T1_LOCOMOTION_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr
    assert e is not None
    for n in names:
        T1_ACTION_SCALE[n] = 0.25 * e / s

# Compute ACTION_SCALE dictionary for full body
T1_FULLBODY_ACTION_SCALE: dict[str, float] = {}
for a in T1_FULLBODY_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr
    assert e is not None
    for n in names:
        T1_FULLBODY_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
    robot = Entity(T1_ROBOT_CFG)
    model = robot.spec.compile()
    viewer.launch(model)
