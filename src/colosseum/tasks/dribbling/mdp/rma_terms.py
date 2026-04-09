"""Concrete RMA encoder terms for the dribbling task.

BallRmaTerm pairs two encoders for the ball latent slot:
  privileged_encoder  — GT ball obs (pos, vel) → MLP → latent  (Phase 1 target)
  adaptation_encoder  — depth frames from head camera → CNN+LSTM → latent  (Phase 2)

Both encoders share latent_dim so Phase 2 training can regress the adaptation
encoder against the frozen privileged encoder output.

Phase 1 (no camera required)
-----------------------------
BallRmaTermCfg(privileged_obs_group="privileged_ball", latent_dim=8)

The depth encoder is instantiated but never called; update() is a no-op when
the camera sensor is absent from the scene.

Phase 2 (depth camera in scene)
--------------------------------
BallRmaTermCfg(
    privileged_obs_group="privileged_ball",
    adaptation_obs_group="depth_frames",
    sensor_name="head_depth_camera",
    latent_dim=8,
)

update() rolls the depth buffer each step. encode_adaptation() prefers frames
stored in obs_dict (correct gradient flow from rollout buffer) over the
internal buffer (used at collection time).
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


@dataclass(kw_only=True)
class BallRmaTermCfg(RmaTermCfg):
  """Config for the dribbling ball encoder term."""

  privileged_obs_group: str = "privileged_ball"
  """Obs group fed to the privileged encoder. Must exist in the observation manager."""

  adaptation_obs_group: str | None = None
  """Key under which depth frames are passed to encode_adaptation() during Phase 2.
  If None, encode_adaptation() returns zeros (Phase 1 — encoder unused)."""

  # --- depth encoder architecture ---
  lstm_hidden: int = 64
  """LSTM hidden size inside the depth CNN+LSTM encoder."""

  # --- depth preprocessing (update()) ---
  sensor_name: str = "head_rgbd"
  """Scene sensor name to read raw depth frames from. Phase 2 only."""

  seq_len: int = 5
  """Frames in the rolling temporal buffer."""

  height: int = 72
  """Target frame height after bilinear downsampling."""

  width: int = 128
  """Target frame width after bilinear downsampling."""

  depth_clip: float = 6.0
  """Max depth in metres; frames are clipped then normalised to [0, 1]."""

  def build(self, env: ManagerBasedRlEnv) -> BallRmaTerm:
    return BallRmaTerm(cfg=self, env=env)


class BallRmaTerm(RmaTerm):
  """Privileged MLP + depth CNN+LSTM encoder pair for ball information."""

  def __init__(self, cfg: BallRmaTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)

    # Privileged encoder: infer input dim from the obs manager
    group_dim = env.observation_manager.group_obs_dim[cfg.privileged_obs_group]
    input_dim = group_dim[0] if isinstance(group_dim, tuple) else int(group_dim)
    self._priv_encoder = PrivilegedEncoder(
      input_dim=input_dim, latent_dim=cfg.latent_dim
    ).to(env.device)

    # Adaptation encoder: same latent_dim so Phase 2 target matches exactly
    self._depth_encoder = DepthEncoder(
      latent_dim=cfg.latent_dim, lstm_hidden=cfg.lstm_hidden
    ).to(env.device)

    self._depth_buffer: torch.Tensor | None = None

  # ------------------------------------------------------------------
  # Encoder properties
  # ------------------------------------------------------------------

  @property
  def privileged_encoder(self) -> nn.Module:
    return self._priv_encoder

  @property
  def adaptation_encoder(self) -> nn.Module:
    return self._depth_encoder

  # ------------------------------------------------------------------
  # Encoding
  # ------------------------------------------------------------------

  def encode_privileged(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """GT ball obs → MLP → latent.  (N, latent_dim)"""
    return self._priv_encoder(obs_dict[self.cfg.privileged_obs_group])

  def encode_adaptation(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Depth frames → CNN+LSTM → latent.  (N, latent_dim)

    Reads frames from obs_dict keyed by adaptation_obs_group.
    Phase 2 passes aligned snapshots from get_current_adaptation_obs().
    Falls back to zeros on the very first step before the buffer is populated.
    """
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]

    if cfg.adaptation_obs_group is not None and cfg.adaptation_obs_group in obs_dict:
      return self._depth_encoder(obs_dict[cfg.adaptation_obs_group])

    return torch.zeros(self._env.num_envs, cfg.latent_dim, device=self._env.device)

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def update(self) -> None:
    """Read depth sensor, downsample, normalise, roll buffer.

    No-op when the camera sensor is absent from the scene (Phase 1 training).
    """
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]

    try:
      sensor = self._env.scene[cfg.sensor_name]
    except KeyError:
      return  # camera absent from scene (Phase 1 training)

    depth = sensor.data.depth  # (N, H, W, 1)
    depth = depth.permute(0, 3, 1, 2).float()            # (N, 1, H, W)
    depth = F.interpolate(
      depth,
      size=(cfg.height, cfg.width),
      mode="bilinear",
      align_corners=False,
    )
    depth = depth.clamp(0.0, cfg.depth_clip) / cfg.depth_clip  # [0, 1]

    if self._depth_buffer is None:
      self._depth_buffer = (
        depth.unsqueeze(1).expand(-1, cfg.seq_len, -1, -1, -1).clone()
      )  # (N, seq_len, 1, H, W) — fill with first frame
    else:
      self._depth_buffer = torch.cat(
        [self._depth_buffer[:, 1:], depth.unsqueeze(1)], dim=1
      )  # roll: drop oldest, append newest

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    """Zero depth buffer for reset environments."""
    if self._depth_buffer is not None and env_ids is not None:
      self._depth_buffer[env_ids] = 0.0

  def get_current_adaptation_obs(self) -> dict[str, torch.Tensor]:
    """Return current depth buffer keyed by adaptation_obs_group."""
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]
    if cfg.adaptation_obs_group is None or self._depth_buffer is None:
      return {}
    return {cfg.adaptation_obs_group: self._depth_buffer}
