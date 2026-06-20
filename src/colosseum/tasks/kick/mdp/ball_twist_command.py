"""Command term: walk-toward-the-ball twist, driven by ball body-frame position.

This is the "ball position as twist command" trick (step 0 of the residual
plan): instead of sampling a random velocity command, the twist is computed each
step from the ball's position in the robot body frame so the frozen walk policy
steers toward the ball. It is registered under the name ``twist`` so the velocity
policy's ``generated_commands("twist")`` observation reads it unchanged.

Command convention (body frame, matching the walk policy's training):
    dir   = normalize(ball_xy_b)
    speed = clip(speed_gain * dist, 0, max_speed), zeroed within stop_distance
    vx,vy = dir * speed
    wz    = clip(stiffness * atan2(py, px), +/- max_wz)

All magnitudes stay inside the walk policy's trained command range so the policy
sees an in-distribution setpoint.
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


class BallTwistCommand(CommandTerm):
  """Body-frame twist [vx, vy, wz] pointing the robot at the ball."""

  cfg: BallTwistCommandCfg

  def __init__(self, cfg: BallTwistCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.twist = torch.zeros((env.num_envs, 3), device=env.device)
    self.metrics["ball_distance"] = torch.zeros(env.num_envs, device=env.device)

  @property
  def command(self) -> torch.Tensor:
    return self.twist

  def _ball_xy_b(self) -> torch.Tensor:
    """Ball position relative to robot base in robot body frame. Shape (N, 2)."""
    robot = self._env.scene[self.cfg.robot_entity]
    ball = self._env.scene[self.cfg.ball_entity]
    relative_w = ball.data.root_link_pos_w[:, :3] - robot.data.root_link_pos_w[:, :3]
    quat_w = robot.data.root_link_quat_w
    quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
    return quat_apply(quat_conj, relative_w)[:, :2]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Nothing to sample: the command is a deterministic function of the ball.
    pass

  def _update_command(self) -> None:
    ball_xy_b = self._ball_xy_b()
    dist = ball_xy_b.norm(dim=-1, keepdim=True)
    direction = ball_xy_b / dist.clamp(min=1e-6)

    speed = (self.cfg.speed_gain * dist).clamp(min=0.0, max=self.cfg.max_speed)
    speed = torch.where(dist > self.cfg.stop_distance, speed, torch.zeros_like(speed))
    self.twist[:, :2] = direction * speed

    yaw_err = torch.atan2(ball_xy_b[:, 1], ball_xy_b[:, 0])
    self.twist[:, 2] = (self.cfg.stiffness * yaw_err).clamp(
      min=-self.cfg.max_wz, max=self.cfg.max_wz
    )

  def _update_metrics(self) -> None:
    self.metrics["ball_distance"] = self._ball_xy_b().norm(dim=-1)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return
    robot = self._env.scene[self.cfg.robot_entity]
    base_pos = robot.data.root_link_pos_w[batch]
    quat_w = robot.data.root_link_quat_w[batch : batch + 1]
    twist_b = torch.cat(
      [self.twist[batch, :2], torch.zeros(1, device=self.twist.device)]
    ).unsqueeze(0)
    twist_w = quat_apply(quat_w, twist_b)[0]
    visualizer.add_arrow(
      start=base_pos.cpu().numpy(),
      end=(base_pos + twist_w * 2.0).cpu().numpy(),
      color=(0.2, 0.6, 1.0, 0.9),
      label=f"ball_twist |v|={self.twist[batch, :2].norm():.2f}",
    )


@dataclass(kw_only=True)
class BallTwistCommandCfg(CommandTermCfg):
  """Configuration for BallTwistCommand."""

  class_type: type[CommandTerm] = BallTwistCommand

  robot_entity: str = "robot"
  ball_entity: str = "ball"

  # Command is recomputed every step; nothing is time-resampled.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  # Map ball distance -> forward/lateral speed, clipped to the walk policy's range.
  speed_gain: float = 1.0
  max_speed: float = 1.0
  # Stop commanding motion once the ball is within this radius.
  stop_distance: float = 0.4
  # Yaw controller turning the robot to face the ball.
  stiffness: float = 1.0
  max_wz: float = 1.0
  debug_vis: bool = True

  def build(self, env: ManagerBasedRlEnv) -> BallTwistCommand:
    return BallTwistCommand(self, env)
