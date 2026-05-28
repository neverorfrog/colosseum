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

from dataclasses import dataclass

import torch
import torch.nn as nn
from mjlab.envs import ManagerBasedRlEnv

from colosseum.algorithm.networks.privileged_encoder import PrivilegedEncoder
from colosseum.algorithm.networks.proprio_encoder import ProprioWindowEncoder
from colosseum.managers.rma_manager import RmaTerm, RmaTermCfg


class OdomHead(nn.Module):
  def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(latent_dim, hidden_dim),
      nn.ELU(),
      nn.Linear(hidden_dim, 2),
    )

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    return self.net(z)


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
  odom_obs_group: str = "odom"
  latent_dim: int = 8
  latent_noise_std: float = 0.05
  window_size: int = 50
  hidden_dim: int = 64
  lambda_odom: float = 1.0

  def build(self, env: ManagerBasedRlEnv) -> VelocityRmaTerm:
    return VelocityRmaTerm(cfg=self, env=env)


class _AdaptEncModule(nn.Module):
  """Wrapper around ProprioWindowEncoder + OdomHead for parameter tracking.

  Registers submodules with numeric keys ("0", "1") so state dict keys
  match the old ``ModuleList``-based checkpoints and avoid non-strict
  loading warnings.

  ``forward()`` returns only the latent from ``ProprioWindowEncoder``
  (what the ONNX export wrapper expects). ``OdomHead`` is used internally
  by ``compute_loss()`` during Phase 2 training.
  """

  def __init__(self, adapt_enc: nn.Module, odom_head: nn.Module) -> None:
    super().__init__()
    self.add_module("0", adapt_enc)
    self.add_module("1", odom_head)

  def forward(self, window: torch.Tensor) -> torch.Tensor:
    return self._modules["0"](window)


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

    priv_obs_dim: int = env.observation_manager.group_obs_dim[cfg.privileged_obs_group][
      0
    ]
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

    self._odom_head = OdomHead(
      latent_dim=cfg.latent_dim,
      hidden_dim=cfg.hidden_dim,
    ).to(device)

    self._adapt_enc_module = _AdaptEncModule(self._adapt_enc, self._odom_head).to(device)

    # Rolling buffer: (N, W, D_actor)
    self._window = torch.zeros(
      env.num_envs, cfg.window_size, actor_obs_dim, device=device
    )

  # ------------------------------------------------------------------
  # RmaTerm properties
  # ------------------------------------------------------------------

  @property
  def privileged_group_names(self) -> list[str]:
    return [self.cfg.privileged_obs_group, self.cfg.odom_obs_group]

  @property
  def privileged_encoder(self) -> nn.Module:
    return self._priv_enc

  @property
  def adaptation_encoder(self) -> nn.Module:
    return self._adapt_enc_module

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

  def compute_loss(
    self,
    privileged_obs: dict[str, torch.Tensor],
    adaptation_obs: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor] | None:
    cfg = self.cfg
    env_params = privileged_obs[cfg.privileged_obs_group]  # (B, T, D_priv)
    B, T = env_params.shape[0], env_params.shape[1]

    with torch.no_grad():
      z_priv = self._priv_enc(env_params.reshape(B * T, -1)).reshape(B, T, cfg.latent_dim)

    z_adapt = self.encode_adaptation(adaptation_obs)  # (B, T, latent_dim)
    latent_err = (z_adapt - z_priv).pow(2).mean(dim=-1)  # (B, T)

    gt_vel = privileged_obs[cfg.odom_obs_group]  # (B, T, 3)
    odom_pred = self._odom_head(z_adapt)          # (B, T, 2)
    odom_err = (odom_pred - gt_vel[..., :2]).pow(2).mean(dim=-1)  # (B, T)

    if mask is not None:
      latent_loss = latent_err[mask].mean()
      odom_loss = odom_err[mask].mean()
    else:
      latent_loss = latent_err.mean()
      odom_loss = odom_err.mean()

    return {
      "latent_mse": latent_loss,
      "odom": cfg.lambda_odom * odom_loss,
    }

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
    ids = (
      env_ids
      if not isinstance(env_ids, slice)
      else torch.arange(self._env.num_envs, device=self._window.device)[env_ids]
    )
    if len(ids) == 0:
      return
    obs_buf = self._env.observation_manager.compute()
    actor_obs = obs_buf["actor"]  # (N, D)
    # Broadcast current obs to fill the entire window for each reset env
    self._window[ids] = actor_obs[ids].unsqueeze(1).expand(-1, self.cfg.window_size, -1)
