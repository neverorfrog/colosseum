from mjlab.envs.mdp.dr import body_com_offset, encoder_bias, geom_friction
from mjlab.envs.mdp.events import (
  push_by_setting_velocity,
  reset_joints_by_offset,
  reset_root_state_uniform,
)
# Check if these exist in your mjlab version — they're in task_layer.md docs:
from mjlab.envs.mdp.dr import randomize_rigid_body_mass, randomize_actuator_gains
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.robots.t1_23dof.constants import BASE_BODY_NAME, FOOT_GEOM_NAMES
from colosseum.tasks.velocity.mdp.events import randomize_joint_frictionloss  # custom, see below

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
      "position_range": (-0.05, 0.05),   # was (0.0, 0.0) — FIX 1
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  ),

  # ------------------------------------------------------------------ #
  # STARTUP — persistent per-env physics randomization                  #
  # ------------------------------------------------------------------ #
  "foot_friction": EventTermCfg(
    mode="startup",
    func=geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES),
      "operation": "abs",
      "ranges": (0.3, 1.2),
      "shared_random": True,
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
  # FIX 2: base mass DR
  "base_mass": EventTermCfg(
    mode="startup",
    func=randomize_rigid_body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,)),
      "mass_distribution_params": (0.8, 1.2),  # scale range, holosoma uses add [-1,3]kg
    },
  ),
  # FIX 3: link mass DR (all links except base)
  "link_mass": EventTermCfg(
    mode="startup",
    func=randomize_rigid_body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "mass_distribution_params": (0.9, 1.1),  # conservative: booster_gym ±2%, holosoma [0.9,1.2]
    },
  ),
  # FIX 4: Kp/Kd scale — THE most important missing DR
  "actuator_gains": EventTermCfg(
    mode="startup",
    func=randomize_actuator_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "stiffness_range": (0.9, 1.1),   # holosoma range; widen to (0.85, 1.1) if still vibrating
      "damping_range": (0.9, 1.1),
    },
  ),
  # FIX 5: joint frictionloss — custom function needed
  "joint_friction": EventTermCfg(
    mode="startup",
    func=randomize_joint_frictionloss,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "scale_range": (1.0, 20.0),  # ankle frictionloss=0.1 Nm → 0.1–2.0 Nm
    },
  ),

  # ------------------------------------------------------------------ #
  # INTERVAL — disturbances                                             #
  # ------------------------------------------------------------------ #
  "push_robot": EventTermCfg(
    func=push_by_setting_velocity,
    mode="interval",
    interval_range_s=(5.0, 15.0),   # was (1.0, 3.0) — widened to allow clean gait to develop
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
}




import torch
from mjlab.envs.manager_based_env import ManagerBasedEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def randomize_joint_frictionloss(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Scale joint frictionloss per environment.

    MuJoCo's dof_frictionloss is Coulomb friction in Nm applied at each DOF.
    Scaling the MJCF-defined values (e.g. ankle=0.1 Nm) covers the realistic
    range of stiction in real harmonic-drive joints (0.1–2.0 Nm for ankles).

    Args:
        scale_range: (min, max) uniform scale applied to default frictionloss.
                     (1.0, 20.0) maps ankle's 0.1 Nm → 0.1–2.0 Nm.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    n = len(env_ids)

    # Default frictionloss from MJCF (per joint, not per DOF — assumes 1 DOF/joint)
    default_friction = asset.data.default_joint_frictionloss[env_ids][:, joint_ids]

    lo, hi = scale_range
    scale = torch.rand(n, len(joint_ids), device=env.device) * (hi - lo) + lo

    asset.set_joint_frictionloss(
        default_friction * scale,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )