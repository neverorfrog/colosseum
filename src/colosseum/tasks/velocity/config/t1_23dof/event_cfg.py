from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import (
  apply_body_impulse,
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
  "foot_friction": EventTermCfg(
    mode="reset",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "ranges": {0: (0.2, 2.0), 1: (0.02, 0.05)},
      "axes": [0, 1],
      "operation": "abs",
      "shared_random": True,
    },
  ),
  # ------------------------------------------------------------------ #
  # INTERVAL — disturbances                                            #
  # ------------------------------------------------------------------ #
  # Small, frequent micro-perturbations (always active, no curriculum).
  "kick_robot": EventTermCfg(
    func=push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "velocity_range": {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (-0.1, 0.1),
        "roll": (-0.04, 0.04),
        "pitch": (-0.04, 0.04),
        "yaw": (-0.04, 0.04),
      },
    },
  ),
  # Large, infrequent sustained impacts — magnitude controlled by push_curriculum.
  "push_robot": EventTermCfg(
    func=apply_body_impulse,
    mode="step",
    params={
      "force_range": (0.0, 0.0),
      "torque_range": (0.0, 0.0),
      "duration_s": (0.8, 1.0),
      "cooldown_s": (4.0, 6.0),
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
    },
  ),
  # ------------------------------------------------------------------ #
  # STARTUP — persistent per-env physics randomization                 #
  # ------------------------------------------------------------------ #
  "encoder_bias": EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "bias_range": (-0.015, 0.015),
    },
  ),
  "joint_default_pos": EventTermCfg(
    mode="startup",
    func=dr.joint_default_pos,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (-0.01, 0.01),
      "operation": "add",
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
  "trunk_inertia": EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
      "alpha_range": (-0.1, 0.1),
      "t_range": (-0.02, 0.02),
    },
  ),
  "link_inertia": EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": (-0.2, 0.2),
      "d1_range": (-0.1, 0.1),
      "d2_range": (-0.15, 0.15),
      "d3_range": (-0.15, 0.15),
    },
  ),
  "joint_friction": EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (1.0, 20.0),
      "operation": "scale",
    },
  ),
  "joint_damping": EventTermCfg(
    mode="startup",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (0.85, 1.15),
      "operation": "scale",
    },
  ),
  "joint_armature": EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (0.85, 1.15),
      "operation": "scale",
    },
  ),
  "pd_gains": EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.85, 1.15),
      "kd_range": (0.85, 1.15),
      "operation": "scale",
    },
  ),
  "effort_limits": EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "effort_limit_range": (0.75, 1.0),
      "operation": "scale",
    },
  ),
}
