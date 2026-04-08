"""Concrete RMA encoder terms.

PrivilegedRmaTerm  — GT obs group → MLP encoder → latent  (Phase 1)
DepthRmaTerm       — depth sensor → rolling buffer → CNN+LSTM → latent  (Phase 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from colosseum.algorithm.encoders import DepthEncoder, PrivilegedEncoder
from colosseum.managers.rma_manager import RmaTerm, RmaTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Phase 1 — Privileged MLP encoder
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class PrivilegedRmaTermCfg(RmaTermCfg):
  """Config for a GT obs → MLP encoder term."""

  def build(self, env: ManagerBasedRlEnv) -> PrivilegedRmaTerm:
    return PrivilegedRmaTerm(cfg=self, env=env)


class PrivilegedRmaTerm(RmaTerm):
  """Encodes a GT privileged obs group with a small MLP.

  No internal buffer — encode() reads directly from the GT obs dict.
  Trains jointly with the policy via the PPO loss.
  """

  def __init__(self, cfg: PrivilegedRmaTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    group_dim = env.observation_manager.group_obs_dim[cfg.obs_group]
    assert isinstance(group_dim, tuple) or isinstance(group_dim, int)
    input_dim = group_dim[0] if isinstance(group_dim, tuple) else int(group_dim)
    self._encoder = PrivilegedEncoder(
      input_dim=input_dim, latent_dim=cfg.latent_dim
    ).to(env.device)

  @property
  def needs_privileged_obs(self) -> bool:
    return True

  @property
  def encoder(self) -> nn.Module:
    return self._encoder

  def encode(self, privileged_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    return self._encoder(privileged_obs[self.cfg.obs_group])


# ---------------------------------------------------------------------------
# Phase 2 — Depth camera encoder
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DepthRmaTermCfg(RmaTermCfg):
  """Config for a depth sensor → CNN+LSTM encoder term."""

  sensor_name: str = "head_depth_camera"
  """Scene sensor name to read depth frames from."""

  seq_len: int = 5
  """Number of frames in the rolling temporal buffer."""

  height: int = 72
  """Target frame height after downsampling."""

  width: int = 128
  """Target frame width after downsampling."""

  depth_clip: float = 6.0
  """Max depth in metres before normalising to [0, 1]."""

  lstm_hidden: int = 64
  """LSTM hidden size inside the depth encoder."""

  def build(self, env: ManagerBasedRlEnv) -> DepthRmaTerm:
    return DepthRmaTerm(cfg=self, env=env)


class DepthRmaTerm(RmaTerm):
  """Encodes depth frames from a head-mounted camera.

  Maintains a rolling (N, seq_len, 1, H, W) buffer. update() is called
  every env step to append the latest frame. reset() zeros the buffer for
  terminated/truncated envs. encode() ignores the GT obs dict and runs the
  CNN+LSTM on the internal buffer.
  """

  def __init__(self, cfg: DepthRmaTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._encoder = DepthEncoder(
      latent_dim=cfg.latent_dim,
      lstm_hidden=cfg.lstm_hidden,
    ).to(env.device)
    self._depth_buffer: torch.Tensor | None = None

  @property
  def needs_privileged_obs(self) -> bool:
    return False  # uses internal rolling buffer

  @property
  def encoder(self) -> nn.Module:
    return self._encoder

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def update(self) -> None:
    """Read depth sensor, downsample, clip/normalise, roll buffer."""
    cfg: DepthRmaTermCfg = self.cfg  # type: ignore[assignment]
    sensor = self._env.scene[cfg.sensor_name]
    depth = sensor.data.depth  # (N, H_raw, W_raw, 1)

    depth = depth.permute(0, 3, 1, 2)  # → (N, 1, H_raw, W_raw)
    depth = F.interpolate(
      depth,
      size=(cfg.height, cfg.width),
      mode="bilinear",
      align_corners=False,
    )
    depth = depth.clamp(0, cfg.depth_clip) / cfg.depth_clip  # [0, 1]

    if self._depth_buffer is None:
      self._depth_buffer = (
        depth.unsqueeze(1).expand(-1, cfg.seq_len, -1, -1, -1).clone()
      )
    else:
      self._depth_buffer = torch.cat(
        [self._depth_buffer[:, 1:], depth.unsqueeze(1)], dim=1
      )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if self._depth_buffer is not None:
      self._depth_buffer[env_ids] = 0.0

  # ------------------------------------------------------------------
  # Encoding
  # ------------------------------------------------------------------

  def encode(self, privileged_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode the internal depth buffer. privileged_obs is ignored."""
    if self._depth_buffer is None:
      # Buffer not yet populated — return zeros (first step edge case)
      cfg: DepthRmaTermCfg = self.cfg  # type: ignore[assignment]
      return torch.zeros(self._env.num_envs, cfg.latent_dim, device=self._env.device)
    return self._encoder(self._depth_buffer)
