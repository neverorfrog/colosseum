"""Velocity command produced by the Sokoban discrete plan.

Output shape: [N, 7]

  [:, 0:2]  ball  [vx, vy]   — PUSH: push_dir × ball_speed  / MOVE: [0, 0]
  [:, 2]    padding          — always 0
  [:, 3:5]  robot [vx, vy]   — MOVE: body-frame direction × speed / PUSH: [0, 0]
  [:, 5]    robot omega_z    — MOVE: heading P-controller  / PUSH: 0
  [:, 6]    phase            — PUSH: 1.0  / MOVE: 0.0

Backward compatibility
  Existing rewards and observations read ``get_command("ball_vel")[:, :2]`` for the
  ball velocity target.  Registering SokobanCommand under the key ``"ball_vel"``
  keeps that slice unchanged.

Direction conversion
  The Sokoban plan stores directions in grid space (di, dj).
  Grid-to-local: dx = dj, dy = -di  (grid i downward, local y upward).
  Robot commands are then rotated from local (world-like) frame to robot body frame
  using the same heading-correction P-controller as AbstractionVelocityCommand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply

from colosseum.envs.abstraction_based_env import AbstractionBasedEnv
from colosseum.mdp.abstraction.maze.sokoban_grid_abstraction import SokobanGridAbstraction

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class SokobanCommand(CommandTerm):
  """Velocity command driven by the current Sokoban plan step."""

  cfg: SokobanCommandCfg

  def __init__(self, cfg: SokobanCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    assert isinstance(env, AbstractionBasedEnv), "Requires AbstractionBasedEnv"
    self.env: AbstractionBasedEnv = env

    # [N, 7]: ball_vx, ball_vy, pad, robot_vx, robot_vy, omega_z, phase
    self.command_buffer = torch.zeros((env.num_envs, 7), device=env.device)

    self.metrics["ball_speed"] = torch.zeros(env.num_envs, device=env.device)
    self.metrics["robot_speed"] = torch.zeros(env.num_envs, device=env.device)

  @property
  def command(self) -> torch.Tensor:
    return self.command_buffer

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Zero the buffer for reset envs; the abstraction handles plan re-init."""
    self.command_buffer[env_ids] = 0.0

  def _update_command(self) -> None:
    abstraction = self.env.abstraction_manager.get_term(self.cfg.abstraction_name)
    assert isinstance(abstraction, SokobanGridAbstraction)

    # ── Active mask and action type ──────────────────────────────────────────
    active  = abstraction.current_step < abstraction.plan_length  # [N]
    is_push = abstraction._current_action_is_push & active        # [N]
    is_move = ~abstraction._current_action_is_push & active       # [N]

    # ── Grid direction → local (world-like) frame ────────────────────────────
    # Grid: di=row-delta (down), dj=col-delta (right)
    # Local: x = dj, y = -di
    dir_grid = abstraction._current_direction_grid  # [N, 2]: (di, dj)
    dir_local = torch.stack(
      [dir_grid[:, 1], -dir_grid[:, 0]], dim=1
    )  # [N, 2]: (dx, dy) in local frame
    norm = dir_local.norm(dim=1, keepdim=True).clamp(min=1e-6)
    dir_local_unit = dir_local / norm  # [N, 2]

    # ── Ball velocity (PUSH only, local/world frame) ─────────────────────────
    ball_mask = is_push.float().unsqueeze(1)
    self.command_buffer[:, 0:2] = dir_local_unit * self.cfg.ball_speed * ball_mask
    self.command_buffer[:, 2] = 0.0

    # ── Robot velocity (MOVE only, body frame) ───────────────────────────────
    move_mask = is_move.float().unsqueeze(1)  # [N, 1]

    root_quat_w = self.env.scene["robot"].data.root_link_quat_w  # [N, 4]
    quat_conj = torch.cat([root_quat_w[:, :1], -root_quat_w[:, 1:]], dim=-1)

    dir_local_3d = torch.cat(
      [dir_local_unit * move_mask,
       torch.zeros(self.env.num_envs, 1, device=self.env.device)],
      dim=-1,
    )  # [N, 3]
    dir_body_2d = quat_apply(quat_conj, dir_local_3d)[:, :2]  # [N, 2]
    dir_body_2d = dir_body_2d / dir_body_2d.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    fwd = torch.tensor(
      self.cfg.body_forward_axis[:2], device=self.env.device, dtype=torch.float32
    )
    cross_z = fwd[0] * dir_body_2d[:, 1] - fwd[1] * dir_body_2d[:, 0]
    dot     = fwd[0] * dir_body_2d[:, 0] + fwd[1] * dir_body_2d[:, 1]
    heading_error = torch.atan2(cross_z, dot)  # [N]

    alignment_scale = self.cfg.min_alignment_scale + (
      1.0 - self.cfg.min_alignment_scale
    ) * heading_error.abs().cos()
    linear_speed = (self.cfg.robot_speed * alignment_scale).clamp(
      self.cfg.min_velocity, self.cfg.max_velocity
    )
    omega_z = (heading_error * self.cfg.angular_velocity_gain).clamp(
      -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity
    )

    self.command_buffer[:, 3:5] = dir_body_2d * linear_speed.unsqueeze(1) * move_mask
    self.command_buffer[:, 5]   = omega_z * is_move.float()

    # ── Phase flag ───────────────────────────────────────────────────────────
    self.command_buffer[:, 6] = is_push.float()

    # Zero out everything for inactive envs (plan done)
    self.command_buffer[~active] = 0.0

  def _update_metrics(self) -> None:
    self.metrics["ball_speed"]  = self.command_buffer[:, 0:2].norm(dim=-1)
    self.metrics["robot_speed"] = self.command_buffer[:, 3:5].norm(dim=-1)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    abstraction = self.env.abstraction_manager.get_term(self.cfg.abstraction_name)
    assert isinstance(abstraction, SokobanGridAbstraction)

    # Robot position in local frame
    robot = self.env.scene["robot"]
    robot_pos_local = (
      robot.data.root_link_pos_w[batch, :2]
      - self.env.scene.env_origins[batch, :2]
    )
    origin = self.env.scene.env_origins[batch, :2]

    # Ball position in world frame (for arrow origin)
    ball = self.env.scene["ball"]
    ball_pos_local = ball.data.root_link_pos_w[batch, :2] - origin

    import numpy as np

    is_push = bool(self.command_buffer[batch, 6].item()) > 0.5

    if is_push:
      # Draw ball velocity arrow
      ball_vel = self.command_buffer[batch, 0:2].cpu().numpy()
      ball_start = np.array([ball_pos_local[0].item(), ball_pos_local[1].item(), 0.2])
      visualizer.add_arrow(
        start=ball_start,
        end=ball_start + np.array([ball_vel[0], ball_vel[1], 0.0]) * 2.0,
        color=(1.0, 0.5, 0.0, 0.9),
        label=f"push |v|={float(np.linalg.norm(ball_vel)):.2f}",
      )
    else:
      # Draw robot velocity arrow (body frame → world frame)
      robot_vel_body = self.command_buffer[batch, 3:5]
      robot_vel_3d = torch.cat(
        [robot_vel_body, torch.zeros(1, device=self.env.device)]
      ).unsqueeze(0)
      root_quat = robot.data.root_link_quat_w[batch].unsqueeze(0)
      robot_vel_world = quat_apply(root_quat, robot_vel_3d).squeeze(0).cpu().numpy()
      robot_start = np.array([robot_pos_local[0].item(), robot_pos_local[1].item(), 0.3])
      visualizer.add_arrow(
        start=robot_start,
        end=robot_start + robot_vel_world * 2.0,
        color=(0.2, 0.8, 1.0, 0.9),
        label=f"move ω={self.command_buffer[batch, 5].item():.2f}",
      )


@dataclass(kw_only=True)
class SokobanCommandCfg(CommandTermCfg):
  """Configuration for SokobanCommand."""

  class_type: type[CommandTerm] = SokobanCommand
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  debug_vis: bool = True

  abstraction_name: str = "sokoban"

  # Ball speed during PUSH actions [m/s]
  ball_speed: float = 0.5

  # Robot speed during MOVE actions [m/s]
  robot_speed: float = 1.0

  # Heading correction (same params as AbstractionVelocityCommand)
  body_forward_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
  min_velocity: float = 0.1
  max_velocity: float = 2.0
  min_alignment_scale: float = 0.3
  angular_velocity_gain: float = 2.0
  max_angular_velocity: float = 2.0

  def build(self, env: ManagerBasedRlEnv) -> SokobanCommand:
    return SokobanCommand(self, env)
