"""Custom action terms for colosseum."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """JointPositionAction with steady, sub-step-resolved action latency.

  Models the robot's FastDDS command-path transport delay. At every episode
  reset each environment samples a delay (in physics substeps) that is HELD
  CONSTANT for the whole episode — the steady regime real transport exhibits,
  not per-step jitter — so the policy learns phase margin against a sustained
  lag.

  The delay spans up to ``max_delay_steps`` policy steps at substep resolution.
  A sampled delay ``D = q * decimation + r`` applies the command from ``q + 1``
  policy steps ago during the first ``r`` substeps of the control interval, then
  switches to the command from ``q`` steps ago — i.e. the new target lands ``r``
  substeps into the interval. This is the reference (tum-adlr ``t1.py``)
  intrastep delay generalized from one policy step to ``max_delay_steps``.

  Bounds are in POLICY STEPS, resolved to substeps via the env decimation. The
  delay is drawn from ``[min_delay_steps, max_delay_steps) * decimation``; the
  default 0–2 steps ~= 0–40 ms at 50 Hz with decimation 10.
  """

  min_delay_steps: int = 0
  max_delay_steps: int = 2

  def build(self, env: ManagerBasedRlEnv) -> ActionTerm:
    return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
  """Steady, sub-step-resolved action delay (see cfg)."""

  cfg: DelayedJointPositionActionCfg

  def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._decimation = env.cfg.decimation
    self._low = cfg.min_delay_steps * self._decimation
    self._high = cfg.max_delay_steps * self._decimation
    # History of processed targets at policy-step granularity, most-recent first
    # (row 0 = current step k, row d = step k-d). Depth max_delay_steps+1 covers
    # the deepest lookup history[q+1] (q <= max_delay_steps-1).
    depth = cfg.max_delay_steps + 1
    self._history = self._offset.unsqueeze(0).repeat(depth, 1, 1)
    # Per-env delay in substeps (sampled per episode) and substep index in step.
    self._delay_substeps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._substep = 0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)
    if env_ids is None:
      env_ids = slice(None)
    n = self.num_envs if isinstance(env_ids, slice) else env_ids.numel()
    # Steady regime: resample the held delay once, at episode start.
    if self._high > self._low:
      self._delay_substeps[env_ids] = torch.randint(
        self._low, self._high, (n,), device=self.device
      )
    else:
      self._delay_substeps[env_ids] = self._low
    # Seed history with the default pose so warmup substeps serve a valid target.
    self._history[:, env_ids] = self._offset[env_ids].unsqueeze(0)

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    # Shift history back one step and insert the new target at row 0.
    self._history = torch.roll(self._history, shifts=1, dims=0)
    self._history[0] = self._processed_actions
    self._substep = 0

  def apply_actions(self) -> None:
    q = self._delay_substeps // self._decimation  # whole policy steps back
    r = self._delay_substeps % self._decimation  # sub-step swap point
    # Older target (q+1 back) before the swap point, newer (q back) after.
    idx = torch.where(self._substep < r, q + 1, q)  # (num_envs,)
    env_ids = torch.arange(self.num_envs, device=self.device)
    target = self._history[idx, env_ids]  # (num_envs, num_targets)
    self._substep += 1
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias, joint_ids=self._target_ids
    )


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
