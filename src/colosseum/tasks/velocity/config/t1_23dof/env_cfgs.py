"""Booster T1 velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from colosseum.robots.t1_23dof.constants import (
    ACTION_SCALE,
    FOOT_GEOM_NAMES,
    get_robot_cfg,
)
from colosseum.robots.t1_23dof.contacts import (
    FEET_GROUND_CONTACT_SENSOR,
    SELF_COLLISION_SENSOR,
)


def booster_t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    # Use T1 robot config from t1_constants
    cfg.scene.entities = {"robot": get_robot_cfg()}

    # Foot sites for observations and rewards
    site_names = ("left_foot", "right_foot")

    # Contact sensors from t1_contacts
    cfg.scene.sensors = (FEET_GROUND_CONTACT_SENSOR, SELF_COLLISION_SENSOR)

    # Enable terrain curriculum
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    # Configure action scale (uniform 0.25 for all joints)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = ACTION_SCALE

    # Viewer configuration
    cfg.viewer.body_name = "Trunk"

    # Command visualization height (T1 standing height)
    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 0.0

    # Configure foot height observation
    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names
    # cfg.observations["critic"].terms.pop("foot_height", None)

    # Configure foot friction event with T1 foot geoms
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_NAMES

    # T1-specific pose standards (all 23 joints)
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}

    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body
        r".*Hip_Pitch": 0.3,
        r".*Hip_Roll": 0.15,
        r".*Hip_Yaw": 0.15,
        r".*Knee_Pitch": 0.35,
        r".*Ankle_Pitch": 0.25,
        r".*Ankle_Roll": 0.1,
        # Waist
        r".*Waist": 0.15,
        # Arms
        r".*Shoulder_Pitch": 0.15,
        r".*Shoulder_Roll": 0.15,
        r".*Elbow_Pitch": 0.15,
        r".*Elbow_Yaw": 0.15,
        # Head
        r".*Head.*": 0.1,
    }

    cfg.rewards["pose"].params["std_running"] = {
        # Lower body
        r".*Hip_Pitch": 0.5,
        r".*Hip_Roll": 0.2,
        r".*Hip_Yaw": 0.2,
        r".*Knee_Pitch": 0.6,
        r".*Ankle_Pitch": 0.35,
        r".*Ankle_Roll": 0.15,
        # Waist
        r".*Waist": 0.2,
        # Arms
        r".*Shoulder_Pitch": 0.5,
        r".*Shoulder_Roll": 0.2,
        r".*Elbow_Pitch": 0.35,
        r".*Elbow_Yaw": 0.2,
        # Head
        r".*Head.*": 0.15,
    }

    # Configure body-based rewards
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

    # Configure foot-site-based rewards
    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # Remove foot-site-dependent rewards since T1 doesn't have foot sites defined
    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards.pop(reward_name, None)

    # Reward weights tuning
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 0.0

    # Self-collision penalty
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": SELF_COLLISION_SENSOR.name},
    )

    # Apply play mode overrides
    if play:
        # Effectively infinite episode length
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def booster_t1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 flat terrain velocity configuration."""
    cfg = booster_t1_rough_env_cfg(play=play)

    # Switch to flat terrain
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum
    assert cfg.curriculum is not None
    assert "terrain_levels" in cfg.curriculum
    del cfg.curriculum["terrain_levels"]

    if play:
        commands = cfg.commands
        assert commands is not None
        twist_cmd = commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg
