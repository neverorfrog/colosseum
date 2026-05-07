"""Gait phase clock command term.

Maintains a two-foot phase vector [φ_left, φ_right] that advances at a
per-env frequency sampled at each episode reset.  The policy observes
[cos(φ_L), cos(φ_R), sin(φ_L), sin(φ_R)] — 4 values that encode the
current expected stance/swing state of each foot.

Phase convention:
  - Walking: left foot starts at φ=0, right foot at φ=π (half-period offset).
  - Standing: both feet are snapped to φ=π every step when ‖cmd_xy‖ < threshold
    AND |ω_z| < threshold. At π the cubic Bézier profile in feet_phase evaluates
    to 0, driving both feet to the ground. The observation becomes [-1, -1, 0, 0].
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
    self._phase = torch.zeros((N, 2), device=env.device)
    self._phase[:, 1] = math.pi
    self._phase_dt = torch.zeros(N, device=env.device)
    self._gate_cmd = cfg.gate_command_name

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """[cos(φ_L), cos(φ_R), sin(φ_L), sin(φ_R)]. Shape (N, 4).

    Standing envs have phase snapped to π in _update_command, so they
    produce [-1, -1, 0, 0] — a stable signal the policy learns to associate
    with both feet on the ground. No gating needed here.
    """
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
    self._phase[env_ids, 0] = 0.0
    self._phase[env_ids, 1] = math.pi

  def _update_command(self) -> None:
    advance = self._phase_dt.unsqueeze(-1)
    standing: torch.Tensor | None = None
    if self._gate_cmd is not None:
      cmd = self._env.command_manager.get_command(self._gate_cmd)
      thr = self.cfg.gate_speed_threshold
      moving = (torch.norm(cmd[:, :2], dim=-1) > thr) | (torch.abs(cmd[:, 2]) > thr)
      advance = advance * moving.float().unsqueeze(-1)
      standing = ~moving
    self._phase += advance
    self._phase = (self._phase + math.pi) % (2.0 * math.pi) - math.pi
    if standing is not None and standing.any():
      self._phase[standing] = math.pi

  def _update_metrics(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    pass


@dataclass(kw_only=True)
class GaitPhaseCommandCfg(CommandTermCfg):
  """Configuration for GaitPhaseCommand."""

  class_type: type[CommandTerm] = GaitPhaseCommand

  gait_freq_range: tuple[float, float] = (1.25, 1.75)

  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  gate_command_name: str | None = None
  gate_speed_threshold: float = 0.05

  def build(self, env: ManagerBasedRlEnv) -> GaitPhaseCommand:
    return GaitPhaseCommand(self, env)
