from mjlab.managers import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.mdp.metrics import (
  ang_vel_error,
  base_tilt,
  feet_air_time,
  feet_clearance,
  feet_contact_force,
  forward_velocity,
  joint_acceleration,
  joint_velocity,
  lin_vel_error,
  root_height,
)

_ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))

# Diagnostic-only metrics (no weight, no gradient). Logged to wandb under
# Episode_Metrics/* as per-episode means in physical units, so the gait can be
# read straight off the plots alongside the Episode_Reward/* curves.
metrics = {
  # ---- Posture / balance ----
  "root_height": MetricsTermCfg(func=root_height),
  "base_tilt": MetricsTermCfg(func=base_tilt),
  # ---- Velocity tracking (physical units) ----
  "forward_velocity": MetricsTermCfg(func=forward_velocity),
  "lin_vel_error": MetricsTermCfg(func=lin_vel_error, params={"command_name": "twist"}),
  "ang_vel_error": MetricsTermCfg(func=ang_vel_error, params={"command_name": "twist"}),
  # ---- Motion smoothness / effort ----
  "joint_acceleration": MetricsTermCfg(
    func=joint_acceleration, params={"asset_cfg": _ALL_JOINTS}
  ),
  "joint_velocity": MetricsTermCfg(
    func=joint_velocity, params={"asset_cfg": _ALL_JOINTS}
  ),
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
