"""Booster T1 robot configuration presets.

Hardware specifications for the Booster T1 humanoid robot.
PD gains are computed from motor specifications following the Unitree G1 method.
"""

from colosseum.deploy.config import RobotConfig, PrepareStateConfig
from colosseum.utils import src_dir

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
    # Computed from motor specs: Kp = I_reflected × ω_n²
    # ω_n = 10Hz × 2π = 62.83 rad/s, I_reflected from motor specs
    joint_stiffness=(
        15.99, 15.99,      # Head (small motors)
        160.61, 160.61, 160.61, 160.61,  # Left arm
        160.61, 160.61, 160.61, 160.61,  # Right arm
        188.76,            # Waist
        206.83, 188.76, 188.76, 251.09, 134.05, 134.05,  # Left leg
        206.83, 188.76, 188.76, 251.09, 134.05, 134.05,  # Right leg
    ),

    # PD Gains - Kd (damping)
    # Computed from motor specs: Kd = 2 × ζ × I_reflected × ω_n
    # ζ = 2.0 (overdamped for stability)
    joint_damping=(
        0.68, 0.68,        # Head
        8.52, 8.52, 8.52, 8.52,  # Left arm
        8.52, 8.52, 8.52, 8.52,  # Right arm
        12.02,             # Waist
        13.17, 12.02, 12.02, 15.98, 8.53, 8.53,  # Left leg
        13.17, 12.02, 12.02, 15.98, 8.53, 8.53,  # Right leg
    ),

    # Default standing pose (MUST match HOME_QPOS from constants.py for correct observations!)
    default_joint_pos=(
        0.0, 0.0,          # Head forward
        0.0, -1.4, 0.0, -0.4,  # Left arm (shoulder roll down, elbow yaw -0.4)
        0.0, 1.4, 0.0, 0.4,    # Right arm (shoulder roll down, elbow yaw +0.4)
        0.0,               # Waist straight
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # Left leg (bent knees for standing)
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # Right leg (bent knees for standing)
    ),

    # Effort limits (Nm) - from motor specifications
    effort_limit=(
        7.0, 7.0,          # Head
        18.0, 18.0, 18.0, 18.0,  # Left arm
        18.0, 18.0, 18.0, 18.0,  # Right arm
        25.0,              # Waist
        45.0, 25.0, 25.0, 60.0, 24.0, 15.0,  # Left leg
        45.0, 25.0, 25.0, 60.0, 24.0, 15.0,  # Right leg
    ),

    # Mechanically coupled joints (ankle pairs)
    parallel_joint_indices=(15, 16, 21, 22),

    # MuJoCo model path
    mjcf_path=str(src_dir() / "robots" / "booster_t1" / "xmls" / "T1_23dof.xml"),

    # Prepare state (safe initialization pose for real robot)
    # Higher gains for quick stabilization during initialization
    prepare_state=PrepareStateConfig(
        stiffness=(
            5.0, 5.0,          # Head (low stiffness for safety)
            40.0, 50.0, 20.0, 20.0,  # Left arm
            40.0, 50.0, 20.0, 20.0,  # Right arm
            350.0,             # Waist (high stiffness)
            350.0, 350.0, 180.0, 350.0, 150.0, 150.0,  # Left leg (high)
            350.0, 350.0, 180.0, 350.0, 150.0, 150.0,  # Right leg (high)
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
        joint_pos=(
            0.0, 0.0,          # Head
            0.0, -1.4, 0.0, 0.0,  # Left arm
            0.0, 1.4, 0.0, 0.0,   # Right arm
            0.0,               # Waist
            -0.15, 0.0, 0.0, 0.2, -0.10, 0.0,  # Left leg (knees bent)
            -0.15, 0.0, 0.0, 0.2, -0.10, 0.0,  # Right leg (knees bent)
        ),
    ),
)


__all__ = ["T1_23DOF_ROBOT_CFG"]
