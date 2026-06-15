"""Diagnostic metrics (Waist + legs)."""

from mjlab.managers import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.mdp.metrics import (
  ang_vel_error,
  base_tilt,
  cmd_ang_vel_yaw,
  cmd_lin_vel_x,
  cmd_lin_vel_y,
  feet_air_time,
  feet_clearance,
  feet_contact_force,
  forward_velocity,
  joint_jerk,
  joint_pos_limits_violation,
  joint_torque,
  joint_vibration,
  lin_vel_error,
  root_height,
)

JOINT_GROUPS = {
  "ankle": (".*Ankle.*",),
  "knee": (".*Knee.*",),
  "hip": (".*Hip.*",),
  "waist": ("Waist",),
}

metrics = {
  "root_height": MetricsTermCfg(func=root_height),
  "base_tilt": MetricsTermCfg(func=base_tilt),
  "cmd_lin_vel_x": MetricsTermCfg(func=cmd_lin_vel_x, params={"command_name": "twist"}),
  "cmd_lin_vel_y": MetricsTermCfg(func=cmd_lin_vel_y, params={"command_name": "twist"}),
  "cmd_ang_vel_yaw": MetricsTermCfg(
    func=cmd_ang_vel_yaw, params={"command_name": "twist"}
  ),
  "forward_velocity": MetricsTermCfg(func=forward_velocity),
  "lin_vel_error": MetricsTermCfg(func=lin_vel_error, params={"command_name": "twist"}),
  "ang_vel_error": MetricsTermCfg(func=ang_vel_error, params={"command_name": "twist"}),
  "feet_air_time": MetricsTermCfg(
    func=feet_air_time, params={"sensor_name": "feet_ground_contact"}
  ),
  "feet_clearance": MetricsTermCfg(
    func=feet_clearance, params={"height_sensor_name": "foot_height_scan"}
  ),
  "feet_contact_force": MetricsTermCfg(
    func=feet_contact_force, params={"sensor_name": "feet_ground_contact"}
  ),
}

for _group, _patterns in JOINT_GROUPS.items():
  metrics[f"joint_vibration/{_group}"] = MetricsTermCfg(
    func=joint_vibration,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=_patterns)},
  )
  metrics[f"joint_jerk/{_group}"] = MetricsTermCfg(
    func=joint_jerk,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=_patterns)},
  )
  metrics[f"joint_torque/{_group}"] = MetricsTermCfg(
    func=joint_torque,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=_patterns)},
  )
  metrics[f"joint_pos_limits/{_group}"] = MetricsTermCfg(
    func=joint_pos_limits_violation,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=_patterns)},
  )
