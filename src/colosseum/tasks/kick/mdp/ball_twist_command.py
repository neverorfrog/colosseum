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

  def _to_body(self, rel_xy_w: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame XY vector into the robot body frame. Shape (N, 2)."""
    robot = self._env.scene[self.cfg.robot_entity]
    quat_w = robot.data.root_link_quat_w
    quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
    zeros = torch.zeros(rel_xy_w.shape[0], 1, device=rel_xy_w.device)
    rel_3d = torch.cat([rel_xy_w, zeros], dim=-1)
    return quat_apply(quat_conj, rel_3d)[:, :2]

  def _ball_xy_b(self) -> torch.Tensor:
    """Ball position relative to robot base in robot body frame. Shape (N, 2)."""
    robot = self._env.scene[self.cfg.robot_entity]
    ball = self._env.scene[self.cfg.ball_entity]
    relative_w = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    return self._to_body(relative_w)

  def _dribble_dir(self) -> torch.Tensor:
    """Unit ball->target direction from the ball_vel command (world XY). Zero if undefined."""
    try:
      term = self._env.command_manager.get_term(self.cfg.ball_vel_command_name)
      cmd = term.velocity_command[:, :2]
    except Exception:
      cmd = torch.zeros(self.num_envs, 2, device=self.device)
    norm = cmd.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-6, cmd / norm.clamp(min=1e-6), torch.zeros_like(cmd))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Nothing to sample: the command is a deterministic function of the ball.
    pass

  def _update_command(self) -> None:
    robot = self._env.scene[self.cfg.robot_entity]
    ball = self._env.scene[self.cfg.ball_entity]
    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]

    # Circumnavigation: aim the walk at a point *behind* the ball (opposite the
    # dribble target) until the robot is positioned behind it, then commit to the
    # ball itself. w = alignment of robot->ball with the dribble direction.
    dribble_dir = self._dribble_dir()
    to_ball = ball_xy - robot_xy
    to_ball_dir = to_ball / to_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    w = (to_ball_dir * dribble_dir).sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
    goal_xy = ball_xy - (1.0 - w) * self.cfg.approach_offset * dribble_dir
    goal_b = self._to_body(goal_xy - robot_xy)

    dist = goal_b.norm(dim=-1, keepdim=True)
    direction = goal_b / dist.clamp(min=1e-6)
    speed = (self.cfg.speed_gain * dist).clamp(min=0.0, max=self.cfg.max_speed)
    # Stop based on proximity to the ball (not the goal) so the robot halts in
    # contact range and lets the residual take over the push/kick.
    ball_dist = self._ball_xy_b().norm(dim=-1, keepdim=True)
    speed = torch.where(
      ball_dist > self.cfg.stop_distance, speed, torch.zeros_like(speed)
    )
    self.twist[:, :2] = direction * speed

    yaw_err = torch.atan2(goal_b[:, 1], goal_b[:, 0])
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
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  speed_gain: float = 1.0
  max_speed: float = 1.0
  stop_distance: float = 0.4
  ball_vel_command_name: str = "ball_vel"
  approach_offset: float = 0.0
  stiffness: float = 1.0
  max_wz: float = 1.0
  debug_vis: bool = True

  def build(self, env: ManagerBasedRlEnv) -> BallTwistCommand:
    return BallTwistCommand(self, env)
