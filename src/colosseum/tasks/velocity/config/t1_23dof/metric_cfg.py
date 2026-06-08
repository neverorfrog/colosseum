from mjlab.managers import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.mdp.metrics import (
  ang_vel_error,
  base_tilt,
  commanded_velocity,
  feet_air_time,
  feet_clearance,
  feet_contact_force,
  forward_velocity,
  joint_jerk,
  joint_torque,
  joint_vibration,
  lin_vel_error,
  root_height,
)

# Joint name patterns per group (mirrors the T1 actuator groupings).
JOINT_GROUPS = {
  "ankle": (".*Ankle.*",),
  "knee": (".*Knee.*",),
  "hip": (".*Hip.*",),
  "waist": ("Waist",),
  "arms": (".*Shoulder.*", ".*Elbow.*"),
  "head": ("AAHead_yaw", "Head_pitch"),
}

# Diagnostic-only metrics (no weight, no gradient). Logged to wandb under
# Episode_Metrics/* as per-episode means in physical units, so the gait can be
# read straight off the plots alongside the Episode_Reward/* curves.
metrics = {
  # ---- Posture / balance ----
  "root_height": MetricsTermCfg(func=root_height),
  "base_tilt": MetricsTermCfg(func=base_tilt),
  # ---- Velocity tracking (physical units) ----
  "commanded_velocity": MetricsTermCfg(
    func=commanded_velocity, params={"command_name": "twist"}
  ),
  "forward_velocity": MetricsTermCfg(func=forward_velocity),
  "lin_vel_error": MetricsTermCfg(func=lin_vel_error, params={"command_name": "twist"}),
  "ang_vel_error": MetricsTermCfg(func=ang_vel_error, params={"command_name": "twist"}),
  # ---- Gait / foot kinematics ----
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

# ---- Per-group effort: vibration (mean |joint acc|), jerk (mean |d acc / dt|),
#      and torque (mean |actuator force|) ----
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
