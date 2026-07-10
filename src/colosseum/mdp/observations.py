"""Layer 1: Universal observation functions (robot-agnostic).

These pure functions work across all robots and tasks, depending only on
torch and math utilities. They can be used in both training and deployment.

Also contains navigation-flavored mjlab wrappers (agent_pos, goal_pos, etc.)
that are generic across any task with a navigating robot and a goal command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity, EntityData
from mjlab.managers import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from colosseum.utils.isaaclab import math as lab_math


# ---------------------------------------------------------------------------
# Layer 1: pure math functions (no env dependency)
# ---------------------------------------------------------------------------


def compute_projected_gravity(
    root_quat_w: torch.Tensor,
    gravity_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project gravity vector into base frame (works for any robot).

    This tells the robot which direction is "down" in its own coordinate frame.

    Args:
        root_quat_w: Base orientation quaternion in world frame (w, x, y, z).
                    Shape: (4,) for single instance or (N, 4) for batched.
        gravity_w: Optional gravity vector in world frame. Defaults to [0, 0, -1].
                  Shape: (3,) or (N, 3).

    Returns:
        Gravity vector projected into base frame.
        Shape: (3,) for single instance or (N, 3) for batched.

    Examples:
        >>> # Single instance (deployment)
        >>> quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity (upright)
        >>> gravity = compute_projected_gravity(quat)
        >>> # gravity ≈ [0, 0, -1] (pointing down in base frame)

        >>> # Batched (training)
        >>> quats = torch.randn(4096, 4)  # 4096 environments
        >>> gravities = compute_projected_gravity(quats)
        >>> gravities.shape
        torch.Size([4096, 3])
    """
    if gravity_w is None:
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)

    # Handle both batched and unbatched
    if root_quat_w.dim() == 1:
        # Single instance (deployment)
        return lab_math.quat_apply_inverse(root_quat_w, gravity_w)
    else:
        # Batched (training)
        batch_size = root_quat_w.shape[0]
        if gravity_w.dim() == 1:
            gravity_w = gravity_w.unsqueeze(0).expand(batch_size, -1)
        return lab_math.quat_apply_inverse(root_quat_w, gravity_w)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vector by inverse of quaternion (re-export from isaaclab.math).

    Args:
        quat: Quaternion (w, x, y, z). Shape: (4,) or (N, 4).
        vec: Vector to rotate. Shape: (3,) or (N, 3).

    Returns:
        Rotated vector. Shape matches input.
    """
    return lab_math.quat_apply_inverse(quat, vec)


def base_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Base link height above the terrain floor (env origin Z). Returns [num_envs, 1]."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height.unsqueeze(-1)


def terrain_clearance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Vertical clearance of a TerrainHeightSensor's frame(s) above the terrain.

  Generic reader over ``sensor.data.heights``; the sensor's ``frame`` decides
  what is measured (feet, base, ...). Returns [B, F] (or [B, F, N] if the sensor
  uses ``reduction="none"``)."""
  return env.scene[sensor_name].data.heights


def gait_clock(
  env: ManagerBasedRlEnv,
  command_name: str = "gait_phase",
  twist_command_name: str = "twist",
  speed_threshold: float = 0.05,
) -> torch.Tensor:
  """booster_gym's single-clock gait observation: [cos φ, sin φ]. Returns [N, 2].

  Reads the left-foot phase from the 4D GaitPhaseCommand and gates it to zero
  for standing envs (‖cmd_xy‖ and |ω_z| both below ``speed_threshold``), matching
  booster's ``cos/sin * (gait_freq > 0)`` gate.
  """
  gait = env.command_manager.get_command(command_name)  # [cos_L, cos_R, sin_L, sin_R]
  cos_phi = gait[:, 0:1]
  sin_phi = gait[:, 2:3]
  cmd = env.command_manager.get_command(twist_command_name)
  moving = (torch.norm(cmd[:, :2], dim=-1, keepdim=True) > speed_threshold) | (
    cmd[:, 2:3].abs() > speed_threshold
  )
  gate = moving.float()
  return torch.cat([cos_phi * gate, sin_phi * gate], dim=-1)


# ---------------------------------------------------------------------------
# Privileged critic observations
# ---------------------------------------------------------------------------

def base_external_force(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
) -> torch.Tensor:
  """External force currently applied to the base body (booster push_force). [N, 3]."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_external_force[:, asset_cfg.body_ids[0]]


def base_external_torque(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
) -> torch.Tensor:
  """External torque currently applied to the base body (booster push_torque). [N, 3]."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_external_torque[:, asset_cfg.body_ids[0]]


_BASE_NOMINAL_CACHE: dict = {}


def base_mass_com_offset(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
) -> torch.Tensor:
  """Base com-offset (3) and mass-offset (1) from the nominal model (booster
  base_mass_scaled). Per-env perturbed model field minus the nominal compiled
  value, so it reflects exactly the startup DR applied to the base body. [N, 4]."""
  asset: Entity = env.scene[asset_cfg.name]
  gid = int(asset.indexing.body_ids[asset_cfg.body_ids[0]])
  model = env.sim.model
  # Nominal (pre-DR) values from the compiled MjModel, cached per (model, body).
  key = (id(env.sim.mj_model), gid)
  nominal = _BASE_NOMINAL_CACHE.get(key)
  if nominal is None:
    mjm = env.sim.mj_model
    dev, dt = model.body_mass.device, model.body_mass.dtype
    com_def = torch.as_tensor(mjm.body_ipos[gid], device=dev, dtype=dt)
    mass_def = torch.as_tensor(float(mjm.body_mass[gid]), device=dev, dtype=dt)
    nominal = (com_def, mass_def)
    _BASE_NOMINAL_CACHE[key] = nominal
  com_def, mass_def = nominal
  com_delta = model.body_ipos[:, gid, :] - com_def
  mass_delta = (model.body_mass[:, gid] - mass_def).unsqueeze(-1)
  return torch.cat([com_delta, mass_delta], dim=-1)


__all__ = [
  "base_height",
  "compute_projected_gravity",
  "quat_apply_inverse",
  "agent_pos",
  "agent_pos_local",
  "agent_vel",
  "agent_vel_body",
  "agent_z_vel",
  "goal_pos",
  "goal_pos_local",
  "agent_to_goal_vector",
]


# ---------------------------------------------------------------------------
# Navigation wrappers: generic robot-pos / goal-pos observations
# (robot-agnostic, goal read from command manager "goal" key)
# ---------------------------------------------------------------------------


def agent_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names="root_site"),
) -> torch.Tensor:
  """Agent position in world coordinates. Returns [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  site_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids]
  if site_pos_w.ndim == 3:
    site_pos_w = site_pos_w[:, 0]
  return site_pos_w[:, :2]


def agent_pos_local(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names="root_site"),
) -> torch.Tensor:
  """Agent position in environment-local (maze-centered) coordinates. Returns [num_envs, 2]."""
  return agent_pos(env, asset_cfg) - env.scene.env_origins[:, :2]


def agent_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names="root_site"),
) -> torch.Tensor:
  """Agent XY velocity in world frame. Returns [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  data: EntityData = asset.data
  site_vel_w = data.site_vel_w[:, asset_cfg.site_ids, :2]
  if site_vel_w.ndim == 3:
    site_vel_w = site_vel_w[:, 0]
  return site_vel_w


def agent_vel_body(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Agent linear velocity in body frame (XY). Returns [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, :2]


def agent_z_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names="root_site"),
) -> torch.Tensor:
  """Agent vertical velocity. Returns [num_envs, 1]."""
  asset: Entity = env.scene[asset_cfg.name]
  site_vel_w = asset.data.site_vel_w[:, asset_cfg.site_ids, 2]
  if site_vel_w.ndim == 2:
    site_vel_w = site_vel_w[:, 0]
  return site_vel_w.unsqueeze(-1)


def goal_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Goal position in world coordinates from command manager. Returns [num_envs, 2]."""
  pos: torch.Tensor = env.command_manager.get_command("goal")
  assert pos is not None, "Command 'goal' not found in command manager"
  return pos


def goal_pos_local(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Goal position in environment-local (maze-centered) coordinates. Returns [num_envs, 2]."""
  return goal_pos(env) - env.scene.env_origins[:, :2]


def agent_to_goal_vector(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Vector from agent to goal (goal - agent) in world frame. Returns [num_envs, 2]."""
  return goal_pos(env) - agent_pos(env, asset_cfg)
