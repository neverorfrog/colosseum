"""Gait phase clock command term.

Maintains a two-foot phase vector [φ_left, φ_right] that advances at a
per-env frequency sampled at each episode reset.  The policy observes
[cos(φ_L), cos(φ_R), sin(φ_L), sin(φ_R)] — 4 values that encode the
current expected stance/swing state of each foot.

Phase convention (same as DeepMind T1 joystick reference):
  - Left foot starts at φ=0,  right foot at φ=π  (half-period offset).
  - Phase is wrapped to [-π, π].
  - κ = (1 + cos(φ)) / 2  ∈ [0, 1] is the smooth stance indicator
    (κ≈1 = full stance, κ≈0 = full swing); used by phase-schedule rewards.
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


class GaitPhaseCommand(CommandTerm):
  """Two-foot gait phase clock."""

  cfg: GaitPhaseCommandCfg

  def __init__(self, cfg: GaitPhaseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    N = env.num_envs
    # [φ_left, φ_right], initialised with half-period offset
    self._phase = torch.zeros((N, 2), device=env.device)
    self._phase[:, 1] = math.pi
    # Per-env phase increment (rad / control step)
    self._phase_dt = torch.zeros(N, device=env.device)

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """[cos(φ_L), cos(φ_R), sin(φ_L), sin(φ_R)]. Shape (N, 4)."""
    return torch.cat([torch.cos(self._phase), torch.sin(self._phase)], dim=-1)

  @property
  def phase(self) -> torch.Tensor:
    """Raw phase angles [φ_left, φ_right]. Shape (N, 2)."""
    return self._phase

  @property
  def stance_indicator(self) -> torch.Tensor:
    """κ = (1 + cos(φ)) / 2 ∈ [0, 1] per foot. Shape (N, 2)."""
    return (1.0 + torch.cos(self._phase)) / 2.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    device = self._env.device
    lo, hi = self.cfg.gait_freq_range
    gait_freq = torch.rand(n, device=device) * (hi - lo) + lo
    self._phase_dt[env_ids] = 2.0 * math.pi * self._env.step_dt * gait_freq
    # Reset to standard initial offset
    self._phase[env_ids, 0] = 0.0
    self._phase[env_ids, 1] = math.pi

  def _update_command(self) -> None:
    self._phase += self._phase_dt.unsqueeze(-1)
    # Wrap to [-π, π]
    self._phase = (self._phase + math.pi) % (2.0 * math.pi) - math.pi

  def _update_metrics(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    pass


@dataclass(kw_only=True)
class GaitPhaseCommandCfg(CommandTermCfg):
  """Configuration for GaitPhaseCommand."""

  class_type: type[CommandTerm] = GaitPhaseCommand

  # Gait frequency sampled per episode (Hz).  Reference: U(1.25, 1.75).
  gait_freq_range: tuple[float, float] = (1.25, 1.75)

  # Never resample mid-episode — frequency is fixed per episode.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  def build(self, env: ManagerBasedRlEnv) -> GaitPhaseCommand:
    return GaitPhaseCommand(self, env)
