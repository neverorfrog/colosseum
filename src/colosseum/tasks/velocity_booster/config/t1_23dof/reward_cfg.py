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
    base_height_penalty,
    feet_distance_penalty,
    flat_orientation,
    foot_orientation_penalty,
    orientation_penalty,
)
from colosseum.tasks.velocity_booster.mdp.rewards import (
    arm_swing,
    feet_slip,
    feet_swing,
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
        weight=2.0,
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
            "running_threshold": 1.5,
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
    # =========================
    # Foot contact-outcome rewards (booster_gym style)
    # =========================
    "feet_swing": RewardTermCfg(
        func=feet_swing,
        weight=3.0,
        params={
            "phase_command_name": "gait_phase",
            "sensor_name": "feet_ground_contact",
            "swing_half_width": 0.2 * math.pi,
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    ),
    "arm_swing": RewardTermCfg(
        func=arm_swing,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(
                "Left_Shoulder_Pitch", "Left_Hip_Pitch",
                "Right_Shoulder_Pitch", "Right_Hip_Pitch",
            )),
            "command_name": "twist",
        },
    ),
    "feet_slip": RewardTermCfg(
        func=feet_slip,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
            "sensor_name": "feet_ground_contact",
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
    r"(?i).*shoulder_pitch.*": 0.15,
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
