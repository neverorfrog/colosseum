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

  Maintains an EMA low-pass of the base velocity in the body frame
  (``filtered_lin_vel`` / ``filtered_ang_vel``), mirroring t1.py. This is the
  velocity the error is measured against (via ``_tracking_*_b``) and the velocity
  the filtered tracking rewards read, so the logged error matches what the reward
  sees. Marching-in-place averages to ~0 and so earns neither tracking reward nor
  a low logged error, forcing sustained directed locomotion.
  """

  def __init__(self, cfg: TrueErrorVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._metric_steps = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_y"] = torch.zeros(self.num_envs, device=self.device)
    self.filtered_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
    self.filtered_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

  def _tracking_lin_vel_b(self) -> torch.Tensor:
    return self.filtered_lin_vel

  def _tracking_ang_vel_b(self) -> torch.Tensor:
    return self.filtered_ang_vel

  def reset(self, env_ids):
    self.filtered_lin_vel[env_ids] = 0.0
    self.filtered_ang_vel[env_ids] = 0.0
    return super().reset(env_ids)

  def _update_metrics(self) -> None:
    # Advance the EMA filter before computing the error metrics so error_vel_*
    # measure the same filtered velocity the reward (and any curriculum gate) use.
    fw = self.cfg.filter_weight
    self.filtered_lin_vel.mul_(1.0 - fw).add_(
      self.robot.data.root_link_lin_vel_b, alpha=fw
    )
    self.filtered_ang_vel.mul_(1.0 - fw).add_(
      self.robot.data.root_link_ang_vel_b, alpha=fw
    )
    lin_vel = self._tracking_lin_vel_b()
    ang_vel = self._tracking_ang_vel_b()
    lin_err = self.vel_command_b[:, :2] - lin_vel[:, :2]
    instant_x = torch.abs(lin_err[:, 0])
    instant_y = torch.abs(lin_err[:, 1])
    instant_xy = torch.norm(lin_err, dim=-1)
    instant_yaw = torch.abs(self.vel_command_b[:, 2] - ang_vel[:, 2])
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
  # EMA weight for the base-velocity low-pass: filtered = w*raw + (1-w)*filtered.
  filter_weight: float = 0.10

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

    # The EMA-filtered base velocity (filtered_lin_vel / filtered_ang_vel) is
    # provided by TrueErrorVelocityCommand. The promotion gate (in reset) and the
    # filtered tracking reward read it: marching-in-place averages to ~0 and so
    # earns neither a promotion nor reward, while sustained locomotion does.
    self.metrics["cmd_lin_level"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["cmd_ang_level"] = torch.zeros(self.num_envs, device=self.device)

  def _update_metrics(self) -> None:
    super()._update_metrics()
    if self.cfg.curriculum:
      self.metrics["cmd_lin_level"][:] = self.env_lin.abs().float()
      self.metrics["cmd_ang_level"][:] = self.env_ang.abs().float()

  def reset(self, env_ids):
    # Promotion is evaluated here (episode reset), mirroring t1.py's
    # _update_curriculum call in _reset_idx. mjlab zeroes episode_length_buf
    # *after* command_manager.reset, so it still holds the terminal length, and
    # filtered_*_vel still holds the pre-reset filtered velocity. An env is
    # promoted only if it survived nearly the whole episode while its filtered
    # velocity tracked the command on all three axes. The filter itself is zeroed
    # by super().reset() (TrueErrorVelocityCommand), so read it before that call.
    if self.cfg.curriculum and len(env_ids) > 0:
      max_steps = self._env.max_episode_length
      survived = self._env.episode_length_buf[env_ids].float() > max_steps * (
        1.0 - self.cfg.episode_length_toler
      )
      cmd = self.vel_command_b[env_ids]
      ok = (
        survived
        & (torch.abs(self.filtered_lin_vel[env_ids, 0] - cmd[:, 0]) < self.cfg.x_toler)
        & (torch.abs(self.filtered_lin_vel[env_ids, 1] - cmd[:, 1]) < self.cfg.y_toler)
        & (
          torch.abs(self.filtered_ang_vel[env_ids, 2] - cmd[:, 2]) < self.cfg.yaw_toler
        )
      )
      self._promote(env_ids[ok])
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if not self.cfg.curriculum:
      super()._resample_command(env_ids)
      return

    # Sample new cells from the frontier distribution. Promotion is NOT done
    # here -- it happens once per episode in reset(), as in t1.py. Timed
    # mid-episode resamples just redraw a command from the current frontier.
    flat = torch.multinomial(self.prob.flatten(), len(env_ids), replacement=True)
    lin = (flat // self.n_ang).long() - self.cfg.lin_levels
    ang = (flat % self.n_ang).long() - self.cfg.ang_levels
    self.env_lin[env_ids] = lin
    self.env_ang[env_ids] = ang

    # Cell -> command. Lateral is coupled to lin_level and capped below forward.
    n = len(env_ids)
    jit = lambda: torch.empty(n, device=self.device).uniform_(-0.5, 0.5)  # noqa: E731
    self.vel_command_b[env_ids, 0] = (
      lin.float() + jit()
    ) * self.cfg.lin_vel_x_resolution
    self.vel_command_b[env_ids, 1] = (
      lin.float().abs()
      * torch.empty(n, device=self.device).uniform_(-1.0, 1.0)
      * self.cfg.lin_vel_y_resolution
    )
    self.vel_command_b[env_ids, 2] = (ang.float() + jit()) * self.cfg.ang_vel_resolution

    # Overlay a mutually-exclusive command "style" on the grid command so the
    # policy practices the deployment command distribution (heading-controlled +
    # forward-biased), which it would otherwise never see. All three styles reuse
    # the base UniformVelocityCommand machinery:
    #   standing: is_standing_env -> base _update_command zeroes the command;
    #   forward : straight line, no strafe/turn (vy=wz=0), keeps grid vx;
    #   heading : base _update_command overwrites wz each step with the heading
    #             controller (clip(stiffness * heading_error)); keeps grid vx/vy.
    # The center cell (level 0) is always standing -- a genuine zero command, so
    # the policy learns clean standing there instead of marching in place.
    #
    # Forward and heading envs ride the *forward axis* (env_ang forced to 0) so
    # promotion stays coherent: they only ever promote (lin, 0) cells, since their
    # yaw is 0 (forward) or a heading-derived transient (heading), neither of
    # which corresponds to a non-zero ang_level.
    r = torch.rand(n, device=self.device)
    p_s, p_f = self.cfg.rel_standing_envs, self.cfg.rel_forward_envs
    p_h = self.cfg.rel_heading_envs
    is_center = (lin == 0) & (ang == 0)
    is_standing = is_center | (r < p_s)
    is_forward = (r >= p_s) & (r < p_s + p_f) & ~is_center
    is_heading = (r >= p_s + p_f) & (r < p_s + p_f + p_h) & ~is_center

    fwd_or_head = is_forward | is_heading
    self.env_ang[env_ids[fwd_or_head]] = 0

    fwd_ids = env_ids[is_forward]
    self.vel_command_b[fwd_ids, 1] = 0.0
    self.vel_command_b[fwd_ids, 2] = 0.0

    head_ids = env_ids[is_heading]
    if len(head_ids) > 0:
      assert self.cfg.ranges.heading is not None
      self.heading_target[head_ids] = torch.empty(
        len(head_ids), device=self.device
      ).uniform_(*self.cfg.ranges.heading)

    self.is_standing_env[env_ids] = is_standing
    self.is_forward_env[env_ids] = is_forward
    self.is_heading_env[env_ids] = is_heading

    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    self._reset_metric_accumulators(env_ids)

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
  cell, so the robot first learns to *stand* (a useful bootstrap). Set >=1 to
  skip the standing phase entirely."""
  lin_vel_x_resolution: float = 0.25
  lin_vel_y_resolution: float = 0.10
  ang_vel_resolution: float = 0.20
  # Promotion gate (m/s, rad/s), mirroring t1.py: at episode reset an env is
  # promoted only if it survived ~the whole episode AND its EMA-filtered velocity
  # tracked the command within tolerance on all three axes. Filtering is what
  # forces real walking -- marching-in-place filters to ~0 and so never tracks a
  # non-zero command, regardless of how loose the tolerance is.
  x_toler: float = 0.30
  y_toler: float = 0.20
  yaw_toler: float = 0.20
  update_rate: float = 0.10
  # Episode-survival fraction: episode_length must exceed (1 - this) of the max.
  episode_length_toler: float = 0.10
  # filter_weight (EMA low-pass weight) is inherited from TrueErrorVelocityCommandCfg.

  def build(self, env: ManagerBasedRlEnv) -> CurriculumVelocityCommand:
    return CurriculumVelocityCommand(self, env)
