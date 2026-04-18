"""Velocity command driven by abstraction guidance.

Two movement modes:

  omnidirectional=True  (ball-like robots)
    The robot can move in any direction regardless of its orientation.
    → Output [vx_world, vy_world, 0] in world/maze-local frame directly.

  omnidirectional=False  (legged robots, humanoids)
    The robot has a meaningful body orientation.
    → Output [vx_body, vy_body, ang_vel_z] in body frame.
      The desired direction is rotated from world to body frame, and a
      P-controller on the heading error provides ang_vel_z to align the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from colosseum.envs.abstraction_based_env import AbstractionBasedEnv
from colosseum.mdp.abstraction.maze.grid_abstraction import GridAbstraction
from colosseum.tasks.maze.mdp.observations import agent_pos_local, agent_vel

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class AbstractionVelocityCommand(CommandTerm):
  """Velocity command driven by the grid abstraction direction."""

  cfg: AbstractionVelocityCommandCfg

  def __init__(self, cfg: AbstractionVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    assert isinstance(env, AbstractionBasedEnv), "Requires AbstractionBasedEnv"
    self.env: AbstractionBasedEnv = env

    self.velocity_command = torch.zeros((env.num_envs, 3), device=env.device)
    self._prev_direction = torch.zeros((env.num_envs, 2), device=env.device)
    self._is_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    self.metrics["velocity_magnitude"] = torch.zeros(env.num_envs, device=env.device)

  @property
  def command(self) -> torch.Tensor:
    return self.velocity_command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.velocity_command[env_ids] = 0.0
    self._prev_direction[env_ids] = 0.0
    if self.cfg.rel_standing_envs > 0.0:
      standing = torch.rand(len(env_ids), device=self.env.device) < self.cfg.rel_standing_envs
      self._is_standing[env_ids] = standing

  def _update_command(self) -> None:
    # 1. Query abstraction direction (world/local frame, unit vec)
    query_entity = self.cfg.query_entity
    if self.cfg.use_root_pos:
      entity = self.env.scene[query_entity]
      pos_local = entity.data.root_link_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
    else:
      pos_local = agent_pos_local(
        self.env, SceneEntityCfg(query_entity, site_names=("root_site",))
      )  # [num_envs, 2]

    abstraction = self.env.abstraction_manager.get_term(self.cfg.abstraction_name)
    assert isinstance(abstraction, GridAbstraction)

    direction_local = abstraction.direction(pos_local)  # [num_envs, 2]

    # 2. EMA smoothing to reduce flickering at cell boundaries
    alpha = self.cfg.ema_smoothing
    smoothed = alpha * direction_local + (1.0 - alpha) * self._prev_direction
    norm = torch.linalg.norm(smoothed, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_unit = smoothed / norm  # [num_envs, 2]
    self._prev_direction = direction_unit.clone()

    # 3a. Omnidirectional: output world-frame velocity directly
    if self.cfg.omnidirectional:
      self.velocity_command[:, :2] = direction_unit * self.cfg.base_velocity
      self.velocity_command[:, 2] = 0.0
      return

    # 3b. Heading-constrained (legged robots): body-frame command
    root_quat_w = self.env.scene["robot"].data.root_link_quat_w  # [num_envs, 4]
    quat_conj = torch.cat([root_quat_w[:, :1], -root_quat_w[:, 1:]], dim=-1)
    dir_world_3d = torch.cat(
      [direction_unit, torch.zeros(self.env.num_envs, 1, device=self.env.device)],
      dim=-1,
    )  # [num_envs, 3]
    dir_body_2d = quat_apply(quat_conj, dir_world_3d)[:, :2]  # [num_envs, 2]
    # Re-normalize: xy projection of a 3D rotation loses unit length when body pitches
    dir_body_2d = dir_body_2d / dir_body_2d.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    fwd_2d = torch.tensor(
      self.cfg.body_forward_axis[:2], device=self.env.device, dtype=torch.float32
    )
    cross_z = fwd_2d[0] * dir_body_2d[:, 1] - fwd_2d[1] * dir_body_2d[:, 0]
    dot = fwd_2d[0] * dir_body_2d[:, 0] + fwd_2d[1] * dir_body_2d[:, 1]
    heading_error = torch.atan2(cross_z, dot)  # [num_envs], range [-π, π]

    alignment_scale = self.cfg.min_alignment_scale + (
      1.0 - self.cfg.min_alignment_scale
    ) * torch.abs(torch.cos(heading_error))
    linear_speed = (self.cfg.base_velocity * alignment_scale).clamp(
      self.cfg.min_velocity, self.cfg.max_velocity
    )
    ang_vel = (heading_error * self.cfg.angular_velocity_gain).clamp(
      -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity
    )

    self.velocity_command[:, :2] = dir_body_2d * linear_speed.unsqueeze(-1)
    self.velocity_command[:, 2] = ang_vel

    if self.cfg.rel_standing_envs > 0.0:
      self.velocity_command[self._is_standing] = 0.0

  def _update_metrics(self) -> None:
    self.metrics["velocity_magnitude"] = self.velocity_command[:, :2].norm(dim=-1)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    pos_2d = agent_pos_local(
      self.env, SceneEntityCfg("robot", site_names=("root_site",))
    )[batch]
    pos_3d = (
      torch.cat([pos_2d, torch.tensor([0.3], device=pos_2d.device)]).cpu().numpy()
    )

    vel_2d = self.velocity_command[batch, :2]
    vel_body_3d = torch.cat([vel_2d, torch.zeros(1, device=vel_2d.device)])

    if self.cfg.omnidirectional:
      vel_world_3d = vel_body_3d
    else:
      root_quat_w = self.env.scene["robot"].data.root_link_quat_w[batch].unsqueeze(0)
      vel_world_3d = quat_apply(root_quat_w, vel_body_3d.unsqueeze(0)).squeeze(0)

    vel_world_3d_np = vel_world_3d.cpu().numpy()
    visualizer.add_arrow(
      start=pos_3d,
      end=pos_3d + vel_world_3d_np * 2.0,
      color=(0.1, 0.9, 0.2, 0.8),
      label=f"cmd_vel |v|={vel_2d.norm():.1f}",
    )

    actual_vel_2d = agent_vel(
      self.env, SceneEntityCfg("robot", site_names=("root_site",))
    )[batch]
    actual_vel_3d = (
      torch.cat([actual_vel_2d, torch.zeros(1, device=actual_vel_2d.device)])
      .cpu()
      .numpy()
    )
    visualizer.add_arrow(
      start=pos_3d,
      end=pos_3d + actual_vel_3d * 0.8,
      color=(0.2, 0.2, 0.8, 0.8),
      label="actual_vel",
    )


@dataclass(kw_only=True)
class AbstractionVelocityCommandCfg(CommandTermCfg):
  """Configuration for AbstractionVelocityCommand."""

  class_type: type[CommandTerm] = AbstractionVelocityCommand
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  debug_vis: bool = True

  abstraction_name: str = "grid"

  base_velocity: float = 1.0  # m/s

  # Fraction of envs that receive a zero velocity command each episode (like rel_standing_envs)
  rel_standing_envs: float = 0.0

  # EMA smoothing on the abstraction direction (alpha=1.0 → raw, alpha→0 → heavy)
  ema_smoothing: float = 0.3

  # True  → output [vx_world, vy_world, 0] directly (no heading logic)
  # False → align heading first, then output body-frame [vx, vy, ang_vel_z]
  omnidirectional: bool = False

  # Entity whose position is queried for abstraction direction
  query_entity: str = "robot"

  # If True, use root_link_pos_w instead of root_site position.
  # Set this when the queried entity has no named "root_site" (e.g. a ball).
  use_root_pos: bool = False

  # --- Heading-constrained mode only (omnidirectional=False) ---
  body_forward_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
  min_velocity: float = 0.1
  max_velocity: float = 2.0
  min_alignment_scale: float = 0.3
  angular_velocity_gain: float = 2.0
  max_angular_velocity: float = 2.0

  def build(self, env: ManagerBasedRlEnv) -> AbstractionVelocityCommand:
    return AbstractionVelocityCommand(self, env)
