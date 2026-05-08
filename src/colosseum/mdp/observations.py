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
