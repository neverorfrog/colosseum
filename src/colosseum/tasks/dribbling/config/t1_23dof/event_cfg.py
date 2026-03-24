from mjlab.envs.mdp.dr import body_com_offset, encoder_bias, geom_friction
from mjlab.envs.mdp.events import (
  push_by_setting_velocity,
  reset_joints_by_offset,
  reset_root_state_uniform,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_GEOM_NAMES

events = {
  "reset_ball": EventTermCfg(
    func=reset_root_state_uniform,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("ball"),
      # Random XY offset from env origin; Z fixed at ball radius (on the ground).
      "pose_range": {
        "x": (-3.0, 3.0),
        "y": (-3.0, 3.0),
      },
      "velocity_range": {},
    },
  ),
  "reset_base": EventTermCfg(
    func=reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.01, 0.05),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
    },
  ),
  "reset_robot_joints": EventTermCfg(
    func=reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  ),
  "push_robot": EventTermCfg(
    func=push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      },
    },
  ),
  "foot_friction": EventTermCfg(
    mode="startup",
    func=geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "operation": "abs",
      "ranges": (0.3, 1.2),
      "shared_random": True,  # All foot geoms share the same friction.
    },
  ),
  "encoder_bias": EventTermCfg(
    mode="startup",
    func=encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "bias_range": (-0.015, 0.015),
    },
  ),
  "base_com": EventTermCfg(
    mode="startup",
    func=body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME)),
      "operation": "add",
      "ranges": {
        0: (-0.025, 0.025),
        1: (-0.025, 0.025),
        2: (-0.03, 0.03),
      },
    },
  ),
}
