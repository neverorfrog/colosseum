"""Gait phase clock command term.

Maintains a two-foot phase vector [φ_left, φ_right] that advances at a
per-env frequency sampled at each episode reset.  The policy observes
[cos(φ_L), cos(φ_R), sin(φ_L), sin(φ_R)] — 4 values that encode the
current expected stance/swing state of each foot.

Phase convention:
  - Walking: left foot starts at φ=0 (or random), right foot at φ=π (half-period
    offset from left). When ``randomize_phase=True`` the absolute offset is
    uniformly sampled from [-π, π) so the policy cannot memorise a fixed alignment.
  - Standing: both feet are snapped to φ=π every step when ‖cmd_xy‖ < threshold
    AND |ω_z| < threshold. At π the cubic Bézier profile in feet_phase evaluates
    to 0, driving both feet to the ground. The observation becomes [-1, -1, 0, 0].
  - κ = (1 + cos(φ)) / 2  ∈ [0, 1] is the smooth swing indicator
    (κ≈1 = full swing at φ≈0, κ≈0 = full stance at φ≈π); used by phase-schedule
    rewards.

Phase computation is time-based (not cumulative delta) to avoid floating-point
drift over long episodes::

  φ = episode_length × step_dt × 2π × gait_freq + phase_offset
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
    self._phase_offset = torch.zeros((N, 2), device=env.device)
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
  def swing_indicator(self) -> torch.Tensor:
    """κ = (1 + cos(φ)) / 2 ∈ [0, 1] per foot. Shape (N, 2).

    κ≈1 at φ≈0 (peak swing), κ≈0 at φ≈π (stance / standing snap).
    """
    return (1.0 + torch.cos(self._phase)) / 2.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    device = self._env.device

    cw = self.cfg.gait_freq_center_width
    if cw is not None:
      center, width = float(cw[0]), float(cw[1])
      if width <= 0.0:
        gait_freq = torch.full((n,), center, device=device)
      else:
        gait_freq = torch.rand(n, device=device) * (2 * width) + (center - width)
    else:
      lo, hi = self.cfg.gait_freq_range
      gait_freq = torch.rand(n, device=device) * (hi - lo) + lo

    self._phase_dt[env_ids] = 2.0 * math.pi * self._env.step_dt * gait_freq

    if self.cfg.randomize_phase:
      self._phase_offset[env_ids, 0] = (
        torch.rand(n, device=device) * (2 * math.pi) - math.pi
      )
    else:
      self._phase_offset[env_ids, 0] = 0.0

    self._phase_offset[env_ids, 1] = (
      (self._phase_offset[env_ids, 0] + math.pi + math.pi)
      % (2.0 * math.pi)
      - math.pi
    )

    self._phase[env_ids] = self._phase_offset[env_ids]

  def _update_command(self) -> None:
    episode_len = self._env.episode_length_buf.float().unsqueeze(-1)
    self._phase = episode_len * self._phase_dt.unsqueeze(-1) + self._phase_offset
    self._phase = (self._phase + math.pi) % (2.0 * math.pi) - math.pi

    if self._gate_cmd is not None:
      cmd = self._env.command_manager.get_command(self._gate_cmd)
      thr = self.cfg.gate_speed_threshold
      moving = (torch.norm(cmd[:, :2], dim=-1) > thr) | (torch.abs(cmd[:, 2]) > thr)
      standing = ~moving
      if standing.any():
        self._phase[standing] = math.pi

  def _update_metrics(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    pass


@dataclass(kw_only=True)
class GaitPhaseCommandCfg(CommandTermCfg):
  """Configuration for GaitPhaseCommand.

  Parameters
  ----------
  gait_freq_range : tuple[float, float]
    Uniform frequency range [low, high] in Hz. Ignored if
    ``gait_freq_center_width`` is set. Default ``(1.25, 1.75)``.
  gait_freq_center_width : tuple[float, float] | None
    Symmetric frequency sampling: ``(center_Hz, width_Hz)``. Samples from
    ``[center - width, center + width)``. When set, overrides
    ``gait_freq_range``. Default ``None``.
  randomize_phase : bool
    If True, randomise the initial phase offset at each episode reset so the
    policy learns a phase-agnostic gait. Set to ``False`` for deterministic
    evaluation / play. Default ``True``.
  """

  class_type: type[CommandTerm] = GaitPhaseCommand

  gait_freq_range: tuple[float, float] = (1.25, 1.75)
  gait_freq_center_width: tuple[float, float] | None = None
  randomize_phase: bool = True

  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  gate_command_name: str | None = None
  gate_speed_threshold: float = 0.05

  def build(self, env: ManagerBasedRlEnv) -> GaitPhaseCommand:
    return GaitPhaseCommand(self, env)
