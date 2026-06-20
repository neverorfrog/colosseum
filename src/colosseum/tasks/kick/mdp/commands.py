"""Command term: goal kick angle, expressed in the robot's body frame.

Samples a goal kick angle once per episode as an offset from the robot->ball
direction (WORLD frame, at reset time):

    world_angle = atan2(ball_y - robot_y, ball_x - robot_x) + offset
    offset ~ Uniform(angle_offset_range)

The exposed ``command`` is NOT the world-frame angle: it is that goal direction
re-expressed relative to the robot's current heading,

    command = wrap(world_angle - robot_yaw)

Because robot_yaw changes every step, ``command`` is recomputed every step —
it is robot-state info (where the kick goal currently sits relative to the
robot), not a fixed per-episode value.
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


_VIZ_ARROW_LENGTH = 1.5  # metres — length of the goal-direction arrow in debug view


class BallAngleCommand(CommandTerm):
  """Goal kick direction, expressed in the robot's current body frame.

  ``command`` returns shape (N, 1): the angle (radians) from the robot's
  forward (+x body) axis to the direction the ball should be kicked.
  """

  cfg: BallAngleCommandCfg

  def __init__(self, cfg: BallAngleCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    # Fixed per-episode goal direction, in WORLD frame, shape (N, 1).
    self._world_angle = torch.zeros((env.num_envs, 1), device=env.device)
    # Same goal direction, re-expressed in the robot's current body frame.
    self._command = torch.zeros((env.num_envs, 1), device=env.device)
    # Offset from the robot->ball direction, sampled at reset. Resolved into
    # _world_angle on the first _update_command() after reset: events run
    # before the command manager's reset, so robot/ball world positions are
    # only fresh once _update_command() runs (after sim.forward()).
    self._pending_offset = torch.zeros((env.num_envs, 1), device=env.device)
    self._pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.metrics["angle_deg"] = torch.zeros(env.num_envs, device=env.device)

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Goal kick direction in the robot body frame. Shape (N, 1), radians."""
    return self._command

  @property
  def world_angle(self) -> torch.Tensor:
    """Per-episode goal kick direction in the WORLD frame. Shape (N, 1), radians."""
    return self._world_angle

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    lo, hi = self.cfg.angle_offset_range
    n = len(env_ids)
    offset = torch.empty(n, device=self._env.device).uniform_(lo, hi)
    self._pending_offset[env_ids, 0] = offset
    self._pending[env_ids] = True

  def _update_command(self) -> None:
    robot = self._env.scene[self.cfg.robot_entity]

    if self._pending.any():
      env_ids = self._pending.nonzero(as_tuple=False).squeeze(-1)
      ball = self._env.scene[self.cfg.ball_entity]
      rel_pos = (
        ball.data.root_link_pos_w[env_ids, :2]
        - robot.data.root_link_pos_w[env_ids, :2]
      )
      base_angle = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
      self._world_angle[env_ids, 0] = base_angle + self._pending_offset[env_ids, 0]
      self._pending[env_ids] = False

    quat_w = robot.data.root_link_quat_w
    yaw = torch.atan2(
      2.0 * (quat_w[:, 0] * quat_w[:, 3] + quat_w[:, 1] * quat_w[:, 2]),
      1.0 - 2.0 * (quat_w[:, 2] ** 2 + quat_w[:, 3] ** 2),
    )
    rel = self._world_angle[:, 0] - yaw
    self._command[:, 0] = torch.atan2(torch.sin(rel), torch.cos(rel))

  def _update_metrics(self) -> None:
    self.metrics["angle_deg"] = self._command[:, 0].rad2deg()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return
    import numpy as np

    ball_pos = self._env.scene[self.cfg.ball_entity].data.root_link_pos_w[batch]
    origin = ball_pos.cpu().numpy()

    angle = self._world_angle[batch, 0].item()
    end = np.array([
      origin[0] + math.cos(angle) * _VIZ_ARROW_LENGTH,
      origin[1] + math.sin(angle) * _VIZ_ARROW_LENGTH,
      origin[2],
    ])
    visualizer.add_arrow(
      start=origin,
      end=end,
      color=(1.0, 0.5, 0.0, 0.9),
      width=0.025,
      label=f"ball goal {math.degrees(angle):.1f}°",
    )


@dataclass(kw_only=True)
class BallAngleCommandCfg(CommandTermCfg):
  """Configuration for BallAngleCommand."""

  class_type: type[CommandTerm] = BallAngleCommand

  # Sample once per episode; never resample mid-episode.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  debug_vis: bool = True

  robot_entity: str = "robot"
  ball_entity: str = "ball"

  # Goal kick direction is sampled once per episode as an offset (radians)
  # from the robot->ball direction (world frame, at reset time).
  angle_offset_range: tuple[float, float] = (-math.pi / 4, math.pi / 4)

  def build(self, env: ManagerBasedRlEnv) -> BallAngleCommand:
    return BallAngleCommand(self, env)
