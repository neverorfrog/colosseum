"""Command term: line up behind the ball, then strike straight through it.

Two phases, switched by a per-env latch:

- **Lineup** (default): turn-to-face and walk toward a standoff point behind the
  ball on the ball->goal line, so the frozen walk arcs around to line the kick up.
- **Strike** (latched once the robot is behind the ball): aim at a carrot well
  beyond the ball on the same line and drive toward it at ``max_speed``. The robot
  passes straight through the ball; the position target still corrects lateral
  offset (so it can't sail past), and because the carrot doesn't depend on
  ``cos_behind`` nothing teleports as the robot crosses the ball.

The latch stays set until episode reset, so once committed the robot can't flip
back to lineup the instant it passes the ball.

The kick direction is read from the ``ball_angle`` command term (its per-episode
WORLD-frame ``world_angle``), which is the single source of truth the residual
policy also observes. ``ball_angle`` must therefore be updated *before* this
term each step (order the commands dict accordingly).

    kick_dir   = [cos(world_angle), sin(world_angle)]            (world)
    cos_behind = dot(normalize(robot_xy - ball_xy), -kick_dir)   # +1 when behind
    striking  |= cos_behind >= behind_threshold                  # latched

The twist uses the same body-frame, forward-only convention as the other ball
twist commands so the setpoint stays inside the walk policy's trained range.
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

  from colosseum.tasks.kicking_residual.mdp.commands import BallAngleCommand


class BallStandoffTwistCommand(CommandTerm):
  """Body-frame twist [vx, vy, wz] steering the robot behind the ball, then through it."""

  cfg: BallStandoffTwistCommandCfg

  def __init__(self, cfg: BallStandoffTwistCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.twist = torch.zeros((env.num_envs, 3), device=env.device)
    self.aim_w = torch.zeros((env.num_envs, 2), device=env.device)
    # Latched per-env: once the robot is behind the ball it commits to striking
    # through it (heading-hold) and won't fall back to lineup until episode reset.
    self.striking = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.metrics["ball_distance"] = torch.zeros(env.num_envs, device=env.device)
    self.metrics["behindness"] = torch.zeros(env.num_envs, device=env.device)

  @property
  def command(self) -> torch.Tensor:
    return self.twist

  def _kick_dir_w(self) -> torch.Tensor:
    """Per-episode kick direction (world frame), from the ball_angle command. (N, 2)."""
    angle_cmd: BallAngleCommand = self._env.command_manager.get_term(
      self.cfg.ball_angle_command_name
    )  # type: ignore[assignment]
    world_angle = angle_cmd.world_angle[:, 0]
    return torch.stack([world_angle.cos(), world_angle.sin()], dim=-1)

  def _update_command(self) -> None:
    robot = self._env.scene[self.cfg.robot_entity]
    ball = self._env.scene[self.cfg.ball_entity]
    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]

    kick_dir = self._kick_dir_w()  # (N, 2)

    to_robot = robot_xy - ball_xy
    to_robot = to_robot / to_robot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_behind = (to_robot * (-kick_dir)).sum(dim=-1)  # (N,)

    # Latch: commit to striking once the robot is behind the ball on the kick line.
    # Stays latched until reset so crossing the ball can't flip the command back to
    # lineup (which previously caused the robot to shuttle back and forth).
    self.striking |= cos_behind >= self.cfg.behind_threshold

    # Lineup aims behind the ball (approach along the kick line). Strike aims at a
    # carrot well BEYOND the ball on the same line: the robot drives through the
    # ball at full speed, while the position target still corrects lateral offset
    # so it doesn't sail past. The carrot doesn't depend on cos_behind, so crossing
    # the ball never flips the aim.
    striking = self.striking.unsqueeze(-1)
    self.aim_w = torch.where(
      striking,
      ball_xy + kick_dir * self.cfg.strike_lookahead,
      ball_xy - kick_dir * self.cfg.standoff,
    )

    aim_rel_w = torch.cat(
      [self.aim_w - robot_xy, torch.zeros_like(robot_xy[:, :1])], dim=-1
    )
    quat_w = robot.data.root_link_quat_w
    quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
    aim_b = quat_apply(quat_conj, aim_rel_w)[:, :2]

    dist = aim_b.norm(dim=-1, keepdim=True)
    direction = aim_b / dist.clamp(min=1e-6)
    speed = (self.cfg.speed_gain * dist).clamp(min=0.0, max=self.cfg.max_speed)
    self.twist[:, :1] = speed * direction[:, :1].clamp(min=0.0)
    self.twist[:, 1] = 0.0
    self.twist[:, 2] = (self.cfg.stiffness * torch.atan2(aim_b[:, 1], aim_b[:, 0])).clamp(
      min=-self.cfg.max_wz, max=self.cfg.max_wz
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Nothing to sample (the command is a deterministic function of the ball and
    # the ball_angle goal direction); just clear the strike latch for reset envs.
    self.striking[env_ids] = False

  def _update_metrics(self) -> None:
    robot = self._env.scene[self.cfg.robot_entity]
    ball = self._env.scene[self.cfg.ball_entity]
    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]
    self.metrics["ball_distance"] = (ball_xy - robot_xy).norm(dim=-1)

    kick_dir = self._kick_dir_w()
    to_robot = robot_xy - ball_xy
    to_robot = to_robot / to_robot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    self.metrics["behindness"] = (to_robot * (-kick_dir)).sum(dim=-1)

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
      label=f"standoff_twist |v|={self.twist[batch, :2].norm():.2f}",
    )
    aim_3d = torch.cat([self.aim_w[batch], base_pos[2:3]])
    visualizer.add_sphere(
      center=aim_3d.cpu().numpy(),
      radius=0.12,
      color=(1.0, 0.2, 0.6, 0.8),
      label="aim",
    )


@dataclass(kw_only=True)
class BallStandoffTwistCommandCfg(CommandTermCfg):
  """Configuration for BallStandoffTwistCommand."""

  class_type: type[CommandTerm] = BallStandoffTwistCommand

  robot_entity: str = "robot"
  ball_entity: str = "ball"
  # Command term providing the per-episode WORLD-frame kick direction.
  ball_angle_command_name: str = "ball_angle"

  # Command is recomputed every step; nothing is time-resampled.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  # Standoff distance behind the ball along -kick_dir during lineup (m).
  standoff: float = 0.5
  # Carrot distance beyond the ball along +kick_dir during the strike (m). Keep
  # >= max_speed / speed_gain so the forward command stays at max_speed through
  # contact (otherwise the robot decelerates into the ball).
  strike_lookahead: float = 1.5
  # cos(angle) threshold above which the robot counts as "behind" the ball and
  # latches into strike mode (1.0 = directly behind on the kick line).
  behind_threshold: float = 0.85

  # Map aim distance -> forward speed, clipped to the walk policy's range.
  speed_gain: float = 1.0
  max_speed: float = 1.0
  # Yaw controller turning the robot to face the aim point.
  stiffness: float = 1.0
  max_wz: float = 1.0
  debug_vis: bool = True

  def build(self, env: ManagerBasedRlEnv) -> BallStandoffTwistCommand:
    return BallStandoffTwistCommand(self, env)
