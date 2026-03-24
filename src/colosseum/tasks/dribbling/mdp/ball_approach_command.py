"""Command term that directs the robot toward the ball.

Adapted from iros26.tasks.maze.ball_tracking_velocity_command, simplified to
remove abstraction dependency — the heading target is always the ball itself.

Output: [vx_body, vy_body, ang_vel_z] in robot body frame, same shape as
UniformVelocityCommandCfg so all downstream velocity-tracking rewards work
without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class BallApproachCommand(CommandTerm):
  """Velocity command: move toward ball, face ball.

  At every step, computes the direction from the robot root to the ball root
  in the world XY plane, transforms it to the robot body frame, and outputs
  a scaled (vx, vy, ang_vel_z) command that the standard velocity-tracking
  rewards can consume.

  Speed is modulated by heading alignment: the robot slows down when it is
  facing away from the ball and accelerates as it aligns, which implicitly
  encourages the robot to first rotate, then approach.
  """

  cfg: BallApproachCommandCfg

  def __init__(self, cfg: BallApproachCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.velocity_command = torch.zeros((env.num_envs, 3), device=env.device)
    self._prev_direction = torch.zeros((env.num_envs, 2), device=env.device)
    self.metrics["ball_distance"] = torch.zeros(env.num_envs, device=env.device)

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return self.velocity_command

  @property
  def ball(self) -> torch.Tensor:
    return self.metrics["local_ball_pos"]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Clear command and EMA state for the reset environments."""
    self.velocity_command[env_ids] = 0.0
    self._prev_direction[env_ids] = 0.0

  def _update_command(self) -> None:
    """Recompute velocity command from current robot and ball positions."""
    device = self._env.device
    N = self._env.num_envs

    robot = self._env.scene[self.cfg.robot_entity]
    ball = self._env.scene[self.cfg.ball_entity]

    robot_pos_w = robot.data.root_link_pos_w[:, :2]  # (N, 2) world XY
    ball_pos_w = ball.data.root_link_pos_w[:, :2]  # (N, 2)

    # ---- Direction robot → ball (world frame) ----
    to_ball = ball_pos_w - robot_pos_w  # (N, 2)
    dist = to_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, 1)
    to_ball_unit = to_ball / dist

    # ---- EMA smoothing to reduce jitter ----
    alpha = self.cfg.ema_smoothing
    smoothed = alpha * to_ball_unit + (1.0 - alpha) * self._prev_direction
    smoothed_norm = smoothed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    to_ball_unit = smoothed / smoothed_norm
    self._prev_direction = to_ball_unit.clone()

    # ---- Transform direction to robot body frame ----
    root_quat_w = robot.data.root_link_quat_w  # (N, 4)
    quat_conj = torch.cat([root_quat_w[:, :1], -root_quat_w[:, 1:]], dim=-1)

    move_3d = torch.cat([to_ball_unit, torch.zeros(N, 1, device=device)], dim=-1)
    move_body = quat_apply(quat_conj, move_3d)[:, :2]  # (N, 2)

    # ---- Heading error: angle from forward axis to ball direction in body frame ----
    fwd = torch.tensor(
      self.cfg.body_forward_axis[:2], device=device, dtype=torch.float32
    )
    cross_z = fwd[0] * move_body[:, 1] - fwd[1] * move_body[:, 0]
    dot = fwd[0] * move_body[:, 0] + fwd[1] * move_body[:, 1]
    heading_error = torch.atan2(cross_z, dot)  # (N,)

    # ---- Speed: scale with heading alignment ----
    alignment = self.cfg.min_alignment_scale + (
      1.0 - self.cfg.min_alignment_scale
    ) * torch.abs(torch.cos(heading_error))
    speed = (self.cfg.base_velocity * alignment).clamp(
      self.cfg.min_velocity, self.cfg.max_velocity
    )

    ang_vel = (heading_error * self.cfg.angular_velocity_gain).clamp(
      -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity
    )

    self.velocity_command[:, :2] = move_body * speed.unsqueeze(-1)
    self.velocity_command[:, 2] = ang_vel

  def _update_metrics(self) -> None:
    robot_pos = self._env.scene[self.cfg.robot_entity].data.root_link_pos_w[:, :2]
    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[:, :2]
    self.metrics["ball_distance"] = (ball_pos - robot_pos).norm(dim=-1)
    self.metrics["local_ball_pos"] = ball_pos - robot_pos
    self.metrics["global_ball_pos"] = ball_pos

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    pos_3d = self._env.scene[self.cfg.robot_entity].data.root_link_pos_w[batch]
    vel_2d = self.velocity_command[batch, :2]
    vel_body_3d = torch.cat([vel_2d, torch.zeros(1, device=vel_2d.device)])

    root_quat_w = (
      self._env.scene[self.cfg.robot_entity].data.root_link_quat_w[batch].unsqueeze(0)
    )
    vel_world_3d = quat_apply(root_quat_w, vel_body_3d.unsqueeze(0)).squeeze(0)

    visualizer.add_arrow(
      start=pos_3d,
      end=pos_3d + vel_world_3d * 2.0,
      color=(0.9, 0.5, 0.1, 0.8),
      label=f"ball_cmd |v|={vel_2d.norm():.2f}",
    )


@dataclass(kw_only=True)
class BallApproachCommandCfg(CommandTermCfg):
  """Configuration for BallApproachCommand."""

  class_type: type[CommandTerm] = BallApproachCommand

  # Never time-out: command is recomputed every step from current positions.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  debug_vis: bool = True

  robot_entity: str = "robot"
  ball_entity: str = "ball"

  # EMA smoothing factor for direction vector (higher = more responsive).
  ema_smoothing: float = 0.2

  # T1 body frame: x is forward.
  body_forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

  base_velocity: float = 0.8
  min_velocity: float = 0.1
  max_velocity: float = 1.5

  angular_velocity_gain: float = 2.0
  max_angular_velocity: float = 1.5

  # Fraction of base_velocity to use when robot faces directly away from ball.
  min_alignment_scale: float = 0.1

  def build(self, env: ManagerBasedRlEnv) -> BallApproachCommand:
    return BallApproachCommand(self, env)
