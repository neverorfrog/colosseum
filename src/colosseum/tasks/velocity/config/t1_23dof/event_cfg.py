from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import (
  push_by_setting_velocity,
  reset_joints_by_offset,
  reset_root_state_uniform,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_GEOM_NAMES

events = {
  # ------------------------------------------------------------------ #
  # RESET — initial conditions                                          #
  # ------------------------------------------------------------------ #
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
      "position_range": (-0.1, 0.1),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  ),
  # ------------------------------------------------------------------ #
  # INTERVAL — disturbances                                             #
  # ------------------------------------------------------------------ #
  "push_robot": EventTermCfg(
    func=push_by_setting_velocity,
    mode="interval",
    interval_range_s=(5.0, 10.0),
    params={
      "velocity_range": {
        "x": (-1.0, 1.0),
        "y": (-1.0, 1.0),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      },
    },
  ),
  # ------------------------------------------------------------------ #
  # STARTUP — persistent per-env physics randomization                  #
  # ------------------------------------------------------------------ #
  "foot_friction": EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "ranges": (0.3, 1.2),
      "operation": "abs",
      "shared_random": True,
    },
  ),
  "encoder_bias": EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "bias_range": (-0.015, 0.015),
    },
  ),
  "base_com": EventTermCfg(
    mode="startup",
    func=dr.body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
      "operation": "add",
      "ranges": {
        0: (-0.02, 0.02),
        1: (-0.02, 0.02),
        2: (-0.02, 0.02),
      },
    },
  ),
  "joint_damping": EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (0.9, 1.1),
      "operation": "scale",
    },
  ),
  "joint_friction": EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(".*(Hip|Knee|Ankle|Waist|Head).*",)
      ),
      "ranges": (1.0, 5.0),
      "operation": "scale",
    },
  ),
  "arm_joint_friction": EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*(Shoulder|Elbow).*",)),
      "ranges": (1.0, 5.0),
      "operation": "scale",
    },
  ),
  "joint_armature": EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (0.9, 1.1),
      "operation": "scale",
    },
  ),
  "pd_gains": EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "operation": "scale",
    },
  ),
  "effort_limits": EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "effort_limit_range": (0.8, 1.0),
      "operation": "scale",
    },
  ),
  # link_inertia runs first (all bodies ≈ ±10% mass+inertia), then base_inertia
  # overwrites the base with higher variation (≈ ±20%). Both read from default
  # model values so they do not compound — the second write wins for the base.
  "link_inertia": EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": (-0.05, 0.05),
    },
  ),
  "base_inertia": EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
      "alpha_range": (-0.1, 0.1),
    },
  ),
}
