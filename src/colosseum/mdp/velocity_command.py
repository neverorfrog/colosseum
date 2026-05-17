from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class TrueErrorVelocityCommand(UniformVelocityCommand):
  """UniformVelocityCommand with error metrics fixed to report true episode-mean error.

  The base class divides each step's instantaneous error by max_command_step
  (the number of steps in one command window), causing the reported metric to
  scale with episode_length / command_window. This subclass uses an incremental
  mean so the metric always equals the true mean tracking error (m/s, rad/s)
  regardless of episode length or early termination.
  """

  def __init__(self, cfg: TrueErrorVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._metric_steps = torch.zeros(self.num_envs, device=self.device)

  def _update_metrics(self) -> None:
    instant_xy = torch.norm(
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
    )
    instant_yaw = torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    )
    self._metric_steps += 1
    n = self._metric_steps
    self.metrics["error_vel_xy"] += (instant_xy - self.metrics["error_vel_xy"]) / n
    self.metrics["error_vel_yaw"] += (instant_yaw - self.metrics["error_vel_yaw"]) / n

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._metric_steps[env_ids] = 0
    self.metrics["error_vel_xy"][env_ids] = 0.0
    self.metrics["error_vel_yaw"][env_ids] = 0.0


@dataclass(kw_only=True)
class TrueErrorVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> TrueErrorVelocityCommand:
    return TrueErrorVelocityCommand(self, env)
