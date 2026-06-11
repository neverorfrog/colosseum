"""Custom action terms for colosseum."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


@dataclass(kw_only=True)
class HeadPerturbActionCfg(ActionTermCfg):
  """Random head joint perturbation simulating a separate head policy.

  Periodically picks random yaw/pitch targets and smoothly interpolates
  between them via the PD controller. The main joint position action must
  appear BEFORE this term in the actions dict so its head joint targets
  get overwritten.

  Set ``enabled=False`` for evaluation: head targets are held at zero
  (default pose) so the real head policy owns the joints.
  """

  head_joint_names: tuple[str, ...] = ("AAHead_yaw", "Head_pitch")
  """Joint names to perturb (must match entity joint names)."""

  yaw_range: tuple[float, float] = (-1.0, 1.0)
  """Range for random yaw targets (radians). Keep within MuJoCo joint limits."""

  pitch_range: tuple[float, float] = (-0.2, 0.8)
  """Range for random pitch targets (radians). Keep within MuJoCo joint limits."""

  interval_range: tuple[float, float] = (1.0, 3.0)
  """Seconds between target changes (uniform random per env)."""

  smoothing_duration: float = 0.3
  """Seconds to interpolate from current target to new target."""

  enabled: bool = True
  """If False, writes zero targets (default pose) instead of random ones."""

  def build(self, env: ManagerBasedRlEnv) -> ActionTerm:
    return HeadPerturbAction(self, env)


class HeadPerturbAction(ActionTerm):
  """Overwrites head joint PD targets with randomly varying targets.

  Runs every physics substep AFTER the main joint position action so its
  write to ``joint_pos_target`` for the head joints is the one that reaches
  the actuator layer.
  """

  cfg: HeadPerturbActionCfg

  def __init__(self, cfg: HeadPerturbActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    entity = self._entity

    jids, _ = entity.find_joints(cfg.head_joint_names)
    self._head_ids = torch.tensor(jids, device=self.device, dtype=torch.long)
    self._num_head = len(jids)

    self._target_a = torch.zeros(self.num_envs, self._num_head, device=self.device)
    self._target_b = torch.zeros_like(self._target_a)
    self._ramp = torch.ones(self.num_envs, device=self.device)
    self._timer = torch.zeros(self.num_envs, device=self.device)
    self._resample_timers(slice(None))

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return torch.zeros(self.num_envs, 0, device=self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    pass

  def apply_actions(self) -> None:
    if not self.cfg.enabled:
      self._entity.set_joint_position_target(
        torch.zeros(self.num_envs, self._num_head, device=self.device),
        joint_ids=self._head_ids,
      )
      return

    dt = self._env.physics_dt

    self._timer -= dt
    expired = self._timer <= 0.0

    if expired.any():
      self._target_a[expired] = self._target_b[expired].clone()
      self._randomize(expired)
      self._ramp[expired] = 0.0
      self._resample_timers(expired)

    dur = max(self.cfg.smoothing_duration, 1e-8)
    self._ramp = torch.clamp(self._ramp + dt / dur, max=1.0)

    r = self._ramp.unsqueeze(-1)
    target = self._target_a * (1.0 - r) + self._target_b * r

    self._entity.set_joint_position_target(target, joint_ids=self._head_ids)

  def _randomize(self, env_ids: torch.Tensor) -> None:
    n = env_ids.count_nonzero().item()
    if n == 0:
      return
    yaw_lo, yaw_hi = self.cfg.yaw_range
    pitch_lo, pitch_hi = self.cfg.pitch_range
    yaw = torch.rand(n, device=self.device) * (yaw_hi - yaw_lo) + yaw_lo
    pitch = torch.rand(n, device=self.device) * (pitch_hi - pitch_lo) + pitch_lo
    self._target_b[env_ids] = torch.stack([yaw, pitch], dim=-1)

  def _resample_timers(self, env_ids: torch.Tensor | slice) -> None:
    lo, hi = self.cfg.interval_range
    if isinstance(env_ids, slice):
      n = self.num_envs
    elif env_ids.dtype == torch.bool:
      n = env_ids.count_nonzero().item()
    else:
      n = env_ids.numel()
    self._timer[env_ids] = torch.rand(n, device=self.device) * (hi - lo) + lo

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._target_a[env_ids] = 0.0
    self._target_b[env_ids] = 0.0
    self._ramp[env_ids] = 1.0
    self._resample_timers(env_ids)
