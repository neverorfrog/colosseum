"""T1 23-DOF robot configuration for velocity tracking deployment.

This configuration is specifically for the velocity tracking task and may differ
from other tasks using the same robot.

Based on booster_deploy/robots/booster.py T1_23DOF_CFG
"""

from colosseum.deploy.core.controllers import RobotCfg, PrepareStateCfg
from colosseum.utils import src_dir


T1_23DOF_VELOCITY_ROBOT_CFG = RobotCfg(
    name="Booster_T1_23DOF_Velocity",

    # Real robot joint order (hardware interface)
    joint_names=[
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
    ],

    # Simulation joint order (IsaacLab/mjlab alphabetical-ish order)
    sim_joint_names=[
        "AAHead_yaw",
        "Left_Shoulder_Pitch",
        "Right_Shoulder_Pitch",
        "Waist",
        "Head_pitch",
        "Left_Shoulder_Roll",
        "Right_Shoulder_Roll",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Left_Elbow_Pitch",
        "Right_Elbow_Pitch",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Elbow_Yaw",
        "Right_Elbow_Yaw",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "Left_Knee_Pitch",
        "Right_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Right_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Ankle_Roll",
    ],

    # Body names
    body_names=[
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
    ],

    # PD gains (from booster_deploy)
    joint_stiffness=[
        4.0, 4.0,  # Head
        4.0, 4.0, 4.0, 4.0,  # Left arm
        4.0, 4.0, 4.0, 4.0,  # Right arm
        80.0,  # Waist
        80.0, 80.0, 80.0, 80.0, 30.0, 30.0,  # Left leg
        80.0, 80.0, 80.0, 80.0, 30.0, 30.0,  # Right leg
    ],
    joint_damping=[
        1.0, 1.0,  # Head
        1.0, 1.0, 1.0, 1.0,  # Left arm
        1.0, 1.0, 1.0, 1.0,  # Right arm
        2.0,  # Waist
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # Left leg
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # Right leg
    ],

    # Default pose (from booster_deploy)
    default_joint_pos=[
        0.0, 0.0,  # Head
        0.0, -1.4, 0.0, 0.0,  # Left arm
        0.0, 1.4, 0.0, 0.0,  # Right arm
        0.0,  # Waist
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Left leg
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Right leg
    ],

    # Effort limits (from booster_deploy)
    effort_limit=[
        7.0, 7.0,  # Head
        18.0, 18.0, 18.0, 18.0,  # Left arm
        18.0, 18.0, 18.0, 18.0,  # Right arm
        25.0,  # Waist
        45.0, 25.0, 25.0, 60.0, 24.0, 15.0,  # Left leg
        45.0, 25.0, 25.0, 60.0, 24.0, 15.0,  # Right leg
    ],

    # Parallel joint indices (ankles with mechanical coupling)
    parallel_joint_indices=[15, 16, 21, 22],

    # MuJoCo model path (adjusted for colosseum structure)
    mjcf_path=str(src_dir() / "robots" / "booster_t1" / "xmls" / "T1_23dof.xml"),

    # Prepare state (for real robot initialization, from booster_deploy)
    prepare_state=PrepareStateCfg(
        stiffness=[
            5.0, 5.0,
            40.0, 50.0, 20.0, 20.0,
            40.0, 50.0, 20.0, 20.0,
            350.0,
            350.0, 350.0, 180.0, 350.0, 150.0, 150.0,
            350.0, 350.0, 180.0, 350.0, 150.0, 150.0,
        ],
        damping=[
            0.1, 0.1,
            0.5, 1.5, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2,
            5.0,
            7.5, 7.5, 3.0, 5.5, 2.0, 2.0,
            7.5, 7.5, 3.0, 5.5, 2.0, 2.0,
        ],
        joint_pos=[
            0.0, 0.0,
            0.0, -1.4, 0.0, 0.0,
            0.0, 1.4, 0.0, 0.0,
            0.0,
            -0.15, 0.0, 0.0, 0.2, -0.10, 0.0,
            -0.15, 0.0, 0.0, 0.2, -0.10, 0.0,
        ],
    ),

    # Simulation body order (empty for now, can be filled if needed)
    sim_body_names=[],
)
