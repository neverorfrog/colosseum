"""VelocityRmaTerm — classic RMA adaptation for the velocity locomotion task.

Phase 1: PrivilegedEncoder encodes DR-randomized physics parameters (foot
         friction, base COM, PD gain scales) → latent z.  The policy trains
         with z via PPO.

Phase 2: ProprioWindowEncoder estimates z_hat from the last window_size
         proprioceptive observations via Conv1D.  MSE regression against the
         frozen PrivilegedEncoder target.

Phase 3: Policy fine-tuned with frozen ProprioWindowEncoder predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.networks.privileged_encoder import PrivilegedEncoder
from colosseum.algorithm.networks.proprio_encoder import ProprioWindowEncoder
from colosseum.managers.rma_manager import RmaTerm, RmaTermCfg


@dataclass(kw_only=True)
class VelocityRmaTermCfg(RmaTermCfg):
  """Config for the velocity-task RMA term.

  privileged_obs_group must match a group defined in the observation manager
  (see observation_cfg.py).  adaptation_obs_group is managed internally by
  the term (proprio window buffer) — it does not correspond to an obs group
  in the manager.
  """

  privileged_obs_group: str = "env_params"
  adaptation_obs_group: str = "proprio_window"
  latent_dim: int = 8
  latent_noise_std: float = 0.05
  window_size: int = 50
  hidden_dim: int = 64

  def build(self, env: ManagerBasedRlEnv) -> VelocityRmaTerm:
    return VelocityRmaTerm(cfg=self, env=env)


class VelocityRmaTerm(RmaTerm):
  """RMA term for velocity locomotion task.

  Maintains a (num_envs, window_size, actor_obs_dim) rolling buffer of
  proprioceptive observations.  update() pushes the latest actor obs into
  the buffer every env step.  reset() fills all slots for reset envs with
  the current obs (prevents cross-episode leakage).
  """

  def __init__(self, cfg: VelocityRmaTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    device = env.device

    priv_obs_dim: int = env.observation_manager.group_obs_dim[cfg.privileged_obs_group][0]
    actor_obs_dim: int = env.observation_manager.group_obs_dim["actor"][0]

    self._priv_enc = PrivilegedEncoder(
      input_dim=priv_obs_dim,
      latent_dim=cfg.latent_dim,
      hidden_dim=cfg.hidden_dim,
    ).to(device)

    self._adapt_enc = ProprioWindowEncoder(
      obs_dim=actor_obs_dim,
      latent_dim=cfg.latent_dim,
      window_size=cfg.window_size,
    ).to(device)

    # Rolling buffer: (N, W, D_actor)
    self._window = torch.zeros(
      env.num_envs, cfg.window_size, actor_obs_dim, device=device
    )

  # ------------------------------------------------------------------
  # RmaTerm properties
  # ------------------------------------------------------------------

  @property
  def privileged_encoder(self) -> nn.Module:
    return self._priv_enc

  @property
  def adaptation_encoder(self) -> nn.Module:
    return self._adapt_enc

  # ------------------------------------------------------------------
  # Encoding interface
  # ------------------------------------------------------------------

  def encode_privileged(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return self._priv_enc(obs_dict[self.cfg.privileged_obs_group])

  def encode_adaptation(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    window = obs_dict[self.cfg.adaptation_obs_group]  # (N, W, D) or (N, T, W, D)
    if window.dim() == 4:
      # Phase 2 training: T windows stacked along dim 1
      N, T, W, D = window.shape
      z_flat = self._adapt_enc(window.reshape(N * T, W, D))  # (N*T, latent)
      return z_flat.reshape(N, T, self.cfg.latent_dim)
    return self._adapt_enc(window)

  # ------------------------------------------------------------------
  # Adaptation obs snapshot (called by RmaManager.get_adaptation_obs)
  # ------------------------------------------------------------------

  def get_current_adaptation_obs(self) -> dict[str, torch.Tensor]:
    return {self.cfg.adaptation_obs_group: self._window}

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def update(self) -> None:
    """Roll the proprio window and push the latest actor obs."""
    obs_buf = self._env.observation_manager.compute()
    actor_obs: torch.Tensor = obs_buf["actor"]  # (N, D) — cached, no re-compute
    self._window = torch.roll(self._window, shifts=-1, dims=1)
    self._window[:, -1, :] = actor_obs.detach()

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    """Fill all window slots for reset envs with the current actor obs."""
    if env_ids is None:
      return
    ids = env_ids if not isinstance(env_ids, slice) else torch.arange(
      self._env.num_envs, device=self._window.device
    )[env_ids]
    if len(ids) == 0:
      return
    obs_buf = self._env.observation_manager.compute()
    actor_obs = obs_buf["actor"]  # (N, D)
    # Broadcast current obs to fill the entire window for each reset env
    self._window[ids] = actor_obs[ids].unsqueeze(1).expand(-1, self.cfg.window_size, -1)
