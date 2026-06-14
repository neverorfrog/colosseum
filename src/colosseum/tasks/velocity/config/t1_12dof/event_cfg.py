"""Domain randomization mirroring booster_gym's T1 randomization block."""

from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import (
  apply_body_impulse,
  push_by_setting_velocity,
  reset_joints_by_offset,
  reset_root_state_uniform,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.robots.t1_12dof.constants import BASE_BODY_NAME, FOOT_GEOM_NAMES

events = {
  # ------------------------------ RESET ------------------------------ #
  "reset_base": EventTermCfg(
    func=reset_root_state_uniform,
    mode="reset",
    params={
      # booster: init_base_pos_xy +-1, random yaw, init_base_lin_vel_xy ~0.1.
      "pose_range": {
        "x": (-1.0, 1.0),
        "y": (-1.0, 1.0),
        "z": (0.0, 0.0),
        "yaw": (-3.14159, 3.14159),
      },
      "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    },
  ),
  "reset_robot_joints": EventTermCfg(
    func=reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (-0.05, 0.05),  # booster init_dof_pos std 0.05
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  ),
  "foot_friction": EventTermCfg(
    mode="reset",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "ranges": {0: (0.1, 2.0), 1: (0.02, 0.05)},
      "axes": [0, 1],
      "operation": "abs",
      "shared_random": True,
    },
  ),
  # ---------------------------- DISTURBANCE --------------------------- #
  # booster kick: every 2 s set a small random base velocity.
  "kick_robot": EventTermCfg(
    func=push_by_setting_velocity,
    mode="interval",
    interval_range_s=(2.0, 2.0),
    params={
      "velocity_range": {
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "roll": (-0.02, 0.02),
        "pitch": (-0.02, 0.02),
        "yaw": (-0.02, 0.02),
      },
    },
  ),
  # booster push: every 5 s apply a 1 s force/torque impulse to the base.
  "push_robot": EventTermCfg(
    func=apply_body_impulse,
    mode="step",
    params={
      "force_range": (-10.0, 10.0),
      "torque_range": (-2.0, 2.0),
      "duration_s": (1.0, 1.0),
      "cooldown_s": (4.0, 4.0),
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
    },
  ),
  # ----------------------------- STARTUP ------------------------------ #
  "pd_gains": EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.95, 1.05),  # booster dof_stiffness scale
      "kd_range": (0.95, 1.05),  # booster dof_damping scale
      "operation": "scale",
    },
  ),
  "joint_friction": EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (0.0, 2.0),  # booster dof_friction additive
      "operation": "add",
    },
  ),
  "base_com": EventTermCfg(
    mode="startup",
    func=dr.body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
      "operation": "add",
      "ranges": {0: (-0.1, 0.1), 1: (-0.1, 0.1), 2: (-0.1, 0.1)},
    },
  ),
  "base_mass": EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
      "ranges": (0.8, 1.2),  # booster base_mass scale
      "operation": "scale",
    },
  ),
}
