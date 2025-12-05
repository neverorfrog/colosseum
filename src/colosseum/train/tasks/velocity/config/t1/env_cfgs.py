"""Booster T1 velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from colosseum.robots.booster_t1.t1_constants import (
  T1_ACTION_SCALE,
  T1_ROBOT_CFG,
  T1_FULLBODY_ACTION_SCALE,
)
from colosseum.robots.booster_t1.t1_contacts import (
  FEET_GROUND_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
  T1_FOOT_GEOM_NAMES,
)


def booster_t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  # Use T1 robot config from t1_constants
  cfg.scene.entities = {"robot": T1_ROBOT_CFG}
  
  # Determine number of actuated joints from robot config
  assert T1_ROBOT_CFG.articulation is not None
  num_actuated_joints = sum(
    len(actuator.joint_names_expr) 
    for actuator in T1_ROBOT_CFG.articulation.actuators
  )
  
  # Determine which action scale to use
  action_scale = T1_FULLBODY_ACTION_SCALE if num_actuated_joints > 12 else T1_ACTION_SCALE

  # Foot sites for observations and rewards
  site_names = ("left_foot", "right_foot")

  # Contact sensors from t1_contacts
  cfg.scene.sensors = (FEET_GROUND_CONTACT_SENSOR, SELF_COLLISION_SENSOR)

  # Enable terrain curriculum
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Configure action scale based on number of joints
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = action_scale

  # Viewer configuration
  cfg.viewer.body_name = "Trunk"

  # Command visualization height (T1 standing height)
  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.665

  # Configure foot height observation
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # Configure foot friction event with T1 foot geoms
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = T1_FOOT_GEOM_NAMES

  # T1-specific pose standards - configure joint_names to only track actuated joints
  # This is critical: the default uses ".*" which matches ALL 23 joints,
  # but we only want to track the actuated joints (12 for locomotion, 23 for fullbody)
  if num_actuated_joints > 12:
    # Fullbody: track all joints
    cfg.rewards["pose"].params["asset_cfg"].joint_names = (".*",)
  else:
    # Locomotion: track only leg joints
    cfg.rewards["pose"].params["asset_cfg"].joint_names = (
      ".*Hip_Pitch",
      ".*Hip_Roll", 
      ".*Hip_Yaw",
      ".*Knee_Pitch",
      ".*Ankle_Pitch",
      ".*Ankle_Roll",
    )
  
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  
  # Base walking/running standards for legs
  cfg.rewards["pose"].params["std_walking"] = {
    r".*Hip_Pitch": 0.3,
    r".*Hip_Roll": 0.15,
    r".*Hip_Yaw": 0.15,
    r".*Knee_Pitch": 0.35,
    r".*Ankle_Pitch": 0.25,
    r".*Ankle_Roll": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*Hip_Pitch": 0.5,
    r".*Hip_Roll": 0.2,
    r".*Hip_Yaw": 0.2,
    r".*Knee_Pitch": 0.6,
    r".*Ankle_Pitch": 0.35,
    r".*Ankle_Roll": 0.15,
  }
  
  # If using fullbody (23 joints), add upper body standards
  if num_actuated_joints > 12:
    cfg.rewards["pose"].params["std_walking"].update({
      r".*Neck.*": 0.1,
      r".*Shoulder.*": 0.2,
      r".*Elbow.*": 0.2,
      r".*Wrist.*": 0.2,
      r".*Waist.*": 0.15,
    })
    cfg.rewards["pose"].params["std_running"].update({
      r".*Neck.*": 0.15,
      r".*Shoulder.*": 0.3,
      r".*Elbow.*": 0.3,
      r".*Wrist.*": 0.3,
      r".*Waist.*": 0.2,
    })

  # Configure body-based rewards
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

  # Configure foot-site-based rewards
  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

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
