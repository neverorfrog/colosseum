from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
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

  Also splits the linear error into separate per-axis metrics (``error_vel_x``,
  ``error_vel_y``) alongside the combined ``error_vel_xy`` so forward vs lateral
  tracking can be diagnosed independently.
  """

  def __init__(self, cfg: TrueErrorVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._metric_steps = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_y"] = torch.zeros(self.num_envs, device=self.device)

  def _update_metrics(self) -> None:
    lin_err = self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2]
    instant_x = torch.abs(lin_err[:, 0])
    instant_y = torch.abs(lin_err[:, 1])
    instant_xy = torch.norm(lin_err, dim=-1)
    instant_yaw = torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    )
    self._metric_steps += 1
    n = self._metric_steps
    self.metrics["error_vel_x"] += (instant_x - self.metrics["error_vel_x"]) / n
    self.metrics["error_vel_y"] += (instant_y - self.metrics["error_vel_y"]) / n
    self.metrics["error_vel_xy"] += (instant_xy - self.metrics["error_vel_xy"]) / n
    self.metrics["error_vel_yaw"] += (instant_yaw - self.metrics["error_vel_yaw"]) / n

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._reset_metric_accumulators(env_ids)

  def _reset_metric_accumulators(self, env_ids: torch.Tensor) -> None:
    self._metric_steps[env_ids] = 0
    self.metrics["error_vel_x"][env_ids] = 0.0
    self.metrics["error_vel_y"][env_ids] = 0.0
    self.metrics["error_vel_xy"][env_ids] = 0.0
    self.metrics["error_vel_yaw"][env_ids] = 0.0


@dataclass(kw_only=True)
class TrueErrorVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> TrueErrorVelocityCommand:
    return TrueErrorVelocityCommand(self, env)


class CurriculumVelocityCommand(TrueErrorVelocityCommand):
  """Performance-gated velocity command on a 2D (lin_level, ang_level) grid.

  Instead of a global time-based schedule that widens ranges for everyone at a
  fixed step count, each env occupies a grid cell and commands are drawn from a
  probability distribution over cells (``torch.multinomial``). A cell only gains
  probability once an env *succeeds* at it -- survives a full command window and
  tracks all three axes within tolerance -- so infeasible diagonal / high-speed
  corners self-prune and the distribution expands only to the feasibility
  frontier.

  Lateral velocity is coupled to the *same* ``lin_level`` as forward and capped
  below it (``lin_vel_y_resolution < lin_vel_x_resolution``), so the command
  "max forward AND max lateral simultaneously" is never sampled -- lateral is
  always the minor axis.

  When ``cfg.curriculum`` is False the term falls back to the base uniform-box
  sampling (used for play / eval, where heading commands and fixed ranges are
  wanted).
  """

  cfg: CurriculumVelocityCommandCfg

  def __init__(self, cfg: CurriculumVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.n_lin = 2 * cfg.lin_levels + 1
    self.n_ang = 2 * cfg.ang_levels + 1
    self.prob = torch.zeros(self.n_lin, self.n_ang, device=self.device)
    # Seed forward/backward at +/- seed_lin_level (0 -> the standing center cell).
    s = cfg.seed_lin_level
    self.prob[cfg.lin_levels + s, cfg.ang_levels] = 1.0
    self.prob[cfg.lin_levels - s, cfg.ang_levels] = 1.0
    self.env_lin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.env_ang = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # Window accumulators for the success gate. Unlike self.metrics (which the
    # base CommandTerm.reset() zeroes before _resample_command runs), these are
    # reset only by our own resample, so they always hold the just-ended
    # window's mean when the gate reads them.
    self._win_steps = torch.zeros(self.num_envs, device=self.device)
    self._win_x = torch.zeros(self.num_envs, device=self.device)
    self._win_y = torch.zeros(self.num_envs, device=self.device)
    self._win_yaw = torch.zeros(self.num_envs, device=self.device)
    self._in_reset = False

    self.metrics["cmd_lin_level"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["cmd_ang_level"] = torch.zeros(self.num_envs, device=self.device)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    if not self.cfg.curriculum:
      return
    lin_err = self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2]
    self._win_steps += 1
    self._win_x += torch.abs(lin_err[:, 0])
    self._win_y += torch.abs(lin_err[:, 1])
    self._win_yaw += torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    )
    self.metrics["cmd_lin_level"][:] = self.env_lin.abs().float()
    self.metrics["cmd_ang_level"][:] = self.env_ang.abs().float()

  def reset(self, env_ids):
    # Flag so _resample_command (invoked by super().reset) knows this is an
    # episode reset (interrupted window) and skips success promotion.
    self._in_reset = True
    out = super().reset(env_ids)
    self._in_reset = False
    return out

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if not self.cfg.curriculum:
      super()._resample_command(env_ids)
      return

    # Promote only on timed resamples: a full command window survived without
    # termination. Episode resets (self._in_reset) carry interrupted windows.
    if not self._in_reset:
      steps = self._win_steps[env_ids].clamp(min=1.0)
      mean_x = self._win_x[env_ids] / steps
      mean_y = self._win_y[env_ids] / steps
      mean_yaw = self._win_yaw[env_ids] / steps
      # Absolute gate: standing makes error == command, which exceeds the
      # tolerance for any cell whose command is above it (every moving cell),
      # but passes the center cell (command ~ 0) for the standing bootstrap.
      ok = (
        (self._win_steps[env_ids] >= self.cfg.min_window_steps)
        & (mean_x < self.cfg.x_toler)
        & (mean_y < self.cfg.y_toler)
        & (mean_yaw < self.cfg.yaw_toler)
      )
      self._promote(env_ids[ok])

    # Sample new cells from the frontier distribution.
    flat = torch.multinomial(self.prob.flatten(), len(env_ids), replacement=True)
    lin = (flat // self.n_ang).long() - self.cfg.lin_levels
    ang = (flat % self.n_ang).long() - self.cfg.ang_levels
    self.env_lin[env_ids] = lin
    self.env_ang[env_ids] = ang

    # Cell -> command. Lateral is coupled to lin_level and capped below forward.
    n = len(env_ids)
    jit = lambda: torch.empty(n, device=self.device).uniform_(-0.5, 0.5)  # noqa: E731
    self.vel_command_b[env_ids, 0] = (lin.float() + jit()) * self.cfg.lin_vel_x_resolution
    self.vel_command_b[env_ids, 1] = (
      lin.float().abs()
      * torch.empty(n, device=self.device).uniform_(-1.0, 1.0)
      * self.cfg.lin_vel_y_resolution
    )
    self.vel_command_b[env_ids, 2] = (ang.float() + jit()) * self.cfg.ang_vel_resolution
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    self._reset_metric_accumulators(env_ids)
    self._win_steps[env_ids] = 0.0
    self._win_x[env_ids] = 0.0
    self._win_y[env_ids] = 0.0
    self._win_yaw[env_ids] = 0.0

  def _promote(self, ids: torch.Tensor) -> None:
    if len(ids) == 0:
      return
    lx = self.env_lin[ids] + self.cfg.lin_levels
    ay = self.env_ang[ids] + self.cfg.ang_levels
    for dl, da in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
      li = lx + dl
      ai = ay + da
      m = (li >= 0) & (li < self.n_lin) & (ai >= 0) & (ai < self.n_ang)
      if m.any():
        vals = torch.full((int(m.sum()),), self.cfg.update_rate, device=self.device)
        self.prob.index_put_((li[m], ai[m]), vals, accumulate=True)
    self.prob.clamp_(max=1.0)

  # Checkpoint persistence. The generic CommandManager.state_dict/load_state_dict
  # mechanism (mjlab) drives these; nothing algorithm-specific is hardcoded.

  def state_dict(self) -> dict[str, torch.Tensor]:
    return {
      "prob": self.prob.detach().cpu().clone(),
      "env_lin": self.env_lin.detach().cpu().clone(),
      "env_ang": self.env_ang.detach().cpu().clone(),
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    prob = state.get("prob")
    if prob is not None and tuple(prob.shape) == tuple(self.prob.shape):
      self.prob.copy_(prob.to(self.device))
    else:
      logger.warning("Curriculum grid shape mismatch on load; keeping fresh grid.")
    # env_lin/env_ang are per-env; restore only if num_envs matches.
    for name, buf in (("env_lin", self.env_lin), ("env_ang", self.env_ang)):
      v = state.get(name)
      if v is not None and tuple(v.shape) == tuple(buf.shape):
        buf.copy_(v.to(self.device))


@dataclass(kw_only=True)
class CurriculumVelocityCommandCfg(TrueErrorVelocityCommandCfg):
  curriculum: bool = True
  """Enable grid-based performance-gated sampling. Set False for play/eval to
  fall back to uniform-box sampling from ``ranges``."""
  lin_levels: int = 6
  ang_levels: int = 6
  seed_lin_level: int = 0
  """Forward/backward level(s) seeded into the grid. 0 seeds the center (zero)
  cell, so the robot first learns to *stand* (a useful bootstrap); the relative
  promotion gate below then forces it off the center cell once it can balance,
  because standing can't satisfy a non-zero command. Set >=1 to skip the
  standing phase entirely."""
  lin_vel_x_resolution: float = 0.25
  lin_vel_y_resolution: float = 0.10
  ang_vel_resolution: float = 0.20
  # Absolute mean-error promotion gate (m/s, rad/s) -- physically interpretable
  # and, unlike a relative gate, it does not collapse below the robot's roughly
  # constant tracking-error floor at low speed. Each tolerance must sit *below*
  # its axis's minimum non-zero command so standing (error == command) cannot
  # promote a moving cell, yet *above* the achievable error floor (~0.1 m/s).
  # The center (zero) cell still passes (standing error ~ 0), preserving the
  # standing bootstrap. y/yaw tolerances exceed their level-1 command magnitude
  # so the minor axes don't block promotion while forward walking is learned.
  x_toler: float = 0.10
  y_toler: float = 0.12
  yaw_toler: float = 0.15
  update_rate: float = 0.10
  min_window_steps: int = 50

  def build(self, env: ManagerBasedRlEnv) -> CurriculumVelocityCommand:
    return CurriculumVelocityCommand(self, env)
