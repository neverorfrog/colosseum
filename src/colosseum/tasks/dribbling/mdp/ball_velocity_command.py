"""Command term: desired ball velocity in world frame.

Samples a random 2-D direction and speed at every episode reset.
Output: [vx_world, vy_world, 0.0] — the velocity the ball should achieve.
All DribbleBot-style rewards compare the actual ball velocity against this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class BallVelocityCommand(CommandTerm):
  """World-frame ball velocity command.

  At each episode reset, samples a unit direction uniformly on the circle and
  a speed from ``[speed_min, speed_max]``.  The command is held fixed until the
  next resample, giving the policy a stable target to track.

  The third component is always 0 (no yaw component for the ball).
  """

  cfg: BallVelocityCommandCfg

  def __init__(self, cfg: BallVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.velocity_command = torch.zeros((env.num_envs, 3), device=env.device)
    self.metrics["ball_distance"] = torch.zeros(env.num_envs, device=env.device)
    self.metrics["cmd_ball_vel_error"] = torch.zeros(env.num_envs, device=env.device)

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return self.velocity_command

  @property
  def world_vel_cmd(self) -> torch.Tensor:
    """World-frame XY ball velocity target. Shape (N, 2)."""
    return self.velocity_command[:, :2]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    device = self._env.device

    lo, hi = -self.cfg.heading_range, self.cfg.heading_range
    angles = torch.rand(n, device=device) * (hi - lo) + lo
    lo, hi = self.cfg.speed_range
    speeds = torch.rand(n, device=device) * (hi - lo) + lo

    self.velocity_command[env_ids, 0] = torch.cos(angles) * speeds
    self.velocity_command[env_ids, 1] = torch.sin(angles) * speeds
    self.velocity_command[env_ids, 2] = 0.0

  def _update_command(self) -> None:
    pass  # Command is fixed between resamples.

  def _update_metrics(self) -> None:
    robot_pos = self._env.scene[self.cfg.robot_entity].data.root_link_pos_w[:, :2]
    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[:, :2]
    ball_vel = self._env.scene[self.cfg.ball_entity].data.root_link_lin_vel_w[:, :2]

    self.metrics["ball_distance"] = (ball_pos - robot_pos).norm(dim=-1)
    self.metrics["cmd_ball_vel_error"] = (self.velocity_command[:, :2] - ball_vel).norm(
      dim=-1
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[batch]
    vel_2d = self.velocity_command[batch, :2]
    vel_3d = torch.cat([vel_2d, torch.zeros(1, device=vel_2d.device)])

    visualizer.add_arrow(
      start=ball_pos,
      end=ball_pos + vel_3d * 2.0,
      color=(0.2, 0.8, 0.2, 0.9),
      label=f"ball_cmd |v|={vel_2d.norm():.2f}",
    )


@dataclass(kw_only=True)
class BallVelocityCommandCfg(CommandTermCfg):
  """Configuration for BallVelocityCommand."""

  class_type: type[CommandTerm] = BallVelocityCommand

  # Resample command every 5–10 seconds (like UniformVelocityCommand).
  resampling_time_range: tuple[float, float] = (5.0, 10.0)

  robot_entity: str = "robot"
  ball_entity: str = "ball"

  # Speed sampled uniformly from this range at each reset/resample.
  speed_range: tuple[float, float] = (0.1, 0.1)

  # Half-width of the heading range in radians (0 = straight forward, pi = all directions).
  heading_range: float = math.pi / 8

  def build(self, env: ManagerBasedRlEnv) -> BallVelocityCommand:
    return BallVelocityCommand(self, env)
