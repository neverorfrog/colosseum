"""Booster T1 robot configuration presets.

Hardware specifications for the Booster T1 humanoid robot.
PD gains and effort limits from holosoma T1 29-DOF training configuration.
"""

from colosseum.deploy.config import RobotConfig, PrepareStateConfig
from colosseum.robots.t1_23dof.constants import XML as T1_XML_PATH
# T1 23-DOF Full Body Configuration
# Hardware specs derived from motor specifications (t1_actuators.py)
# PD gains computed using natural frequency method (ω_n = 10Hz, ζ = 2.0)
T1_23DOF_ROBOT_CFG = RobotConfig(
    name="Booster_T1_23DOF",

    # Real robot joint order (hardware interface)
    joint_names=(
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
    ),

    # Simulation joint order (ACTUAL MuJoCo order, NOT alphabetical!)
    # Verified from compiled model - do NOT change without verification script
    sim_joint_names=(
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
    ),

    # Body names (real robot order)
    body_names=(
        "Trunk",
        "H1",
        "H2",
        "AL1",
        "AL2",
        "AL3",
        "left_hand_link",
        "AR1",
        "AR2",
        "AR3",
        "right_hand_link",
        "Waist",
        "Hip_Pitch_Left",
        "Hip_Roll_Left",
        "Hip_Yaw_Left",
        "Shank_Left",
        "Ankle_Cross_Left",
        "left_foot_link",
        "Hip_Pitch_Right",
        "Hip_Roll_Right",
        "Hip_Yaw_Right",
        "Shank_Right",
        "Ankle_Cross_Right",
        "right_foot_link",
    ),

    # Simulation body names (empty for now)
    sim_body_names=(),

    # PD Gains - Kp (stiffness)
    # MUST match training nominal values (100%) - training randomizes ±10% at runtime
    # From external/holosoma/src/holosoma/holosoma/config_values/robot.py (T1 29-DOF)
    # Head=5, Arms=20, Waist=200, Hip/Knee=200, Ankle=50
    joint_stiffness=(
        5.0, 5.0,          # Head (holosoma T1)
        20.0, 20.0, 20.0, 20.0,  # Left arm (holosoma T1)
        20.0, 20.0, 20.0, 20.0,  # Right arm
        200.0,             # Waist (holosoma T1)
        200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Left leg (holosoma T1)
        200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Right leg
    ),

    joint_damping=(
        0.5, 0.5,          # Head (holosoma T1)
        0.5, 0.5, 0.5, 0.5,  # Left arm (holosoma T1)
        0.5, 0.5, 0.5, 0.5,  # Right arm
        5.0,               # Waist (holosoma T1)
        5.0, 5.0, 5.0, 5.0, 3.0, 3.0,  # Left leg (holosoma T1)
        5.0, 5.0, 5.0, 5.0, 3.0, 3.0,  # Right leg
    ),

    # Joint armature (reflected inertia) - MUST match training XML!
    # CRITICAL: All joints use 0.3 in T1_23dof.xml (60x larger than old hardcoded 0.005!)
    # Small armature = robot falls immediately, 0.3 = stable as in training
    joint_armature=(
        0.3, 0.3,          # Head (from T1_23dof.xml)
        0.3, 0.3, 0.3, 0.3,  # Left arm (from T1_23dof.xml)
        0.3, 0.3, 0.3, 0.3,  # Right arm
        0.3,               # Waist (from T1_23dof.xml)
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3,  # Left leg (from T1_23dof.xml)
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3,  # Right leg
    ),

    # Default standing pose (MUST match HOME_QPOS from constants.py for correct observations!)
    # EXACTLY matches training constants.py HOME_QPOS
    # Arm positions from manufacturer's booster_deploy/locomotion.py (Shoulder_Pitch=0.2!)
    default_joint_pos=(
        0.0, 0.0,          # Head forward
        0.2, -1.3, 0.0, -0.5,  # Left arm (manufacturer's deployment values)
        0.2, 1.3, 0.0, 0.5,    # Right arm (manufacturer's deployment values)
        0.0,               # Waist straight
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # Left leg (matches training)
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # Right leg (matches training)
    ),

    # Effort limits (Nm) - MUST EXACTLY match holosoma T1 training values!
    # CRITICAL: Action scaling = action_scale * effort_limit / stiffness
    # From external/holosoma/src/holosoma/holosoma/config_values/robot.py (T1 29-DOF)
    effort_limit=(
        7.0, 7.0,          # Head (holosoma T1)
        18.0, 18.0, 18.0, 18.0,  # Left arm (holosoma T1, NOT 30!)
        18.0, 18.0, 18.0, 18.0,  # Right arm (holosoma T1, NOT 30!)
        30.0,              # Waist (holosoma T1, NOT 40!)
        45.0, 30.0, 30.0, 60.0, 12.0, 12.0,  # Left leg (holosoma T1)
        45.0, 30.0, 30.0, 60.0, 12.0, 12.0,  # Right leg (holosoma T1)
    ),

    # Mechanically coupled joints (ankle pairs)
    parallel_joint_indices=(15, 16, 21, 22),

    # MuJoCo model path
    mjcf_path=str(T1_XML_PATH),

    # Prepare state (safe initialization pose for real robot)
    # Higher gains for quick stabilization during initialization
    prepare_state=PrepareStateConfig(
        stiffness=(
            5.0, 5.0,          # Head (holosoma T1)
            20.0, 20.0, 20.0, 20.0,  # Left arm (holosoma T1)
            20.0, 20.0, 20.0, 20.0,  # Right arm
            200.0,             # Waist (holosoma T1)
            200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Left leg (holosoma T1)
            200.0, 200.0, 200.0, 200.0, 50.0, 50.0,  # Right leg
        ),
        damping=(
            0.1, 0.1,          # Head
            0.5, 1.5, 0.2, 0.2,  # Left arm
            0.5, 1.5, 0.2, 0.2,  # Right arm
            5.0,               # Waist
            7.5, 7.5, 3.0, 5.5, 2.0, 2.0,  # Left leg
            7.5, 7.5, 3.0, 5.5, 2.0, 2.0,  # Right leg
        ),
        # Slightly bent knees for stability during initialization
        # Arms match default pose to ensure smooth prepare→normal transition
        joint_pos=(
            0.0, 0.0,          # Head
            0.2, -1.3, 0.0, -0.5,  # Left arm (matches default!)
            0.2, 1.3, 0.0, 0.5,    # Right arm (matches default!)
            0.0,               # Waist
            -0.15, 0.0, 0.0, 0.2, -0.10, 0.0,  # Left leg (knees slightly less bent)
            -0.15, 0.0, 0.0, 0.2, -0.10, 0.0,  # Right leg (knees slightly less bent)
        ),
    ),
)


__all__ = ["T1_23DOF_ROBOT_CFG"]
