import math

from mjlab.envs.mdp import (
    action_rate_l2,
    is_alive,
    joint_pos_limits,
)
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
    body_angular_velocity_penalty,
    track_angular_velocity,
    track_linear_velocity,
    variable_posture,
)

from colosseum.mdp.rewards import (
    arm_phase,
    base_height_penalty,
    feet_distance_penalty,
    feet_phase,
    flat_orientation,
    foot_orientation_penalty,
    orientation_penalty,
    pose_deviation_penalty,
    static_stance,
)
from colosseum.robots.t1_23dof.constants import (
    BASE_BODY_NAME,
    FOOT_BODY_NAMES,
    FOOT_SITE_NAMES,
)

rewards = {
    # =======================
    # Task Tracking Rewards
    # =======================
    "track_linear_velocity": RewardTermCfg(
        func=track_linear_velocity,
        weight=3.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
        func=track_angular_velocity,
        weight=3.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    # =========================
    # Survival rewards
    # =========================
    "alive": RewardTermCfg(
        func=is_alive,
        weight=0.25,
    ),
    "upright": RewardTermCfg(
        func=flat_orientation,
        weight=1.0,
        params={
            "std": math.sqrt(0.5),
            "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME)),
        },
    ),
    # =========================
    # Regularization penalties
    # =========================
    "penalty_base_height": RewardTermCfg(
        func=base_height_penalty,
        weight=-20.0,
        params={
            "target_height": 0.66,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    ),
    "penalty_body_ang_vel": RewardTermCfg(
        func=body_angular_velocity_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
    ),
    "penalty_orientation": RewardTermCfg(
        func=orientation_penalty,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME))},
    ),
    "penalty_feet_ori": RewardTermCfg(
        func=foot_orientation_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=(FOOT_BODY_NAMES))},
    ),
    "penalty_action_rate": RewardTermCfg(func=action_rate_l2, weight=-1.0),
    "penalty_pose_deviation": RewardTermCfg(
        func=variable_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "command_name": "twist",
            "std_standing": {},
            "std_walking": {},
            "std_running": {},
            "walking_threshold": 0.05,
            "running_threshold": 1.0,
        },
    ),
    "dof_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-1.0),
    "penalty_feet_distance": RewardTermCfg(
        func=feet_distance_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
            "min_dist": 0.15,
        },
    ),
    "static_stance": RewardTermCfg(
        func=static_stance,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=(FOOT_SITE_NAMES)),
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    ),
    "penalty_upper_posture_standing": RewardTermCfg(
        func=pose_deviation_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(r"(?i).*shoulder.*", r"(?i).*elbow.*", r"(?i).*head.*"),
            ),
            "pose_weights": {
                r"(?i).*shoulder_roll.*": 10.0,
                r"(?i).*shoulder_pitch.*": 3.0,
                r"(?i).*elbow.*": 2.0,
                r"(?i).*head.*": 0.2,
            },
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    ),
    # =========================
    # Gait Phase Rewards
    # =========================
    "feet_phase": RewardTermCfg(
        func=feet_phase,
        weight=4.0,
        params={
            "phase_command_name": "gait_phase",
            "height_sensor_name": "foot_height_scan",
            "swing_height": 0.05,
            "tracking_sigma": 0.008,
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    ),
    "arm_phase": RewardTermCfg(
        func=arm_phase,
        weight=0.5,
        params={
            "phase_command_name": "gait_phase",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=("Left_Shoulder_Pitch", "Right_Shoulder_Pitch"),
            ),
            "swing_amplitude": 0.2,
            "tracking_sigma": 0.25,
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    ),
}

rewards["penalty_pose_deviation"].params["std_standing"] = {".*": 0.05}
rewards["penalty_pose_deviation"].params["std_walking"] = {
    r"(?i).*hip_pitch.*": 0.3,
    r"(?i).*hip_roll.*": 0.15,
    r"(?i).*hip_yaw.*": 0.15,
    r"(?i).*knee.*": 0.35,
    r"(?i).*ankle_pitch.*": 0.25,
    r"(?i).*ankle_roll.*": 0.1,
    r"Waist": 0.1,
    r"(?i).*shoulder_pitch.*": 0.4,
    r"(?i).*shoulder_roll.*": 0.15,
    r"(?i).*elbow.*": 0.15,
    r"(?i).*head.*": 0.2,
}
rewards["penalty_pose_deviation"].params["std_running"] = {
    r"(?i).*hip_pitch.*": 0.5,
    r"(?i).*hip_roll.*": 0.2,
    r"(?i).*hip_yaw.*": 0.2,
    r"(?i).*knee.*": 0.6,
    r"(?i).*ankle_pitch.*": 0.35,
    r"(?i).*ankle_roll.*": 0.15,
    r"Waist": 0.2,
    r"(?i).*shoulder_pitch.*": 0.5,
    r"(?i).*shoulder_roll.*": 0.2,
    r"(?i).*elbow.*": 0.35,
    r"(?i).*head.*": 0.3,
}
