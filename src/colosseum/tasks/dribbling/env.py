from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from mjlab.envs import ManagerBasedRlEnvCfg, types

from colosseum.envs.viewer_compatible_env import ViewerCompatibleEnv
from colosseum.tasks.dribbling.mdp.depth_encoder import DepthEncoder, ProjectionHead
from colosseum.tasks.dribbling.viz import draw_camera_ball_overlay

class DribblingEnv(ViewerCompatibleEnv):
  """Dribbling environment with depth encoder producing z_enc."""

  def __init__(self, cfg: DribblingEnvCfg, **kwargs: Any) -> None:
    super().__init__(cfg, **kwargs)

    self._enc_h = cfg.depth_enc_height
    self._enc_w = cfg.depth_enc_width
    self._enc_seq_len = cfg.depth_enc_seq_len
    self._depth_clip = cfg.depth_clip

    self.depth_encoder = DepthEncoder(
      latent_dim=cfg.z_enc_dim,
      input_height=self._enc_h,
      input_width=self._enc_w,
    ).to(self.device)
    self.projection_head = ProjectionHead(latent_dim=cfg.z_enc_dim).to(self.device)

    # Runtime buffers — initialized on first _update_z_enc() call.
    self._depth_buffer: torch.Tensor | None = None  # (N, T, 1, H, W)
    self.z_enc: torch.Tensor | None = None  # (N, z_enc_dim)

  # ------------------------------------------------------------------
  # Depth buffer + encoder update
  # ------------------------------------------------------------------

  def _update_z_enc(self) -> None:
    """Read depth sensor, downsample, buffer, and run encoder."""
    sensor = self.scene["head_rgbd"]
    depth = sensor.data.depth  # (N, H_raw, W_raw, 1)

    # (N, 1, H_raw, W_raw) → bilinear downsample → (N, 1, enc_h, enc_w)
    depth = depth.permute(0, 3, 1, 2)
    depth = F.interpolate(
      depth, size=(self._enc_h, self._enc_w), mode="bilinear", align_corners=False
    )

    # Clip and normalize to [0, 1]
    depth = depth.clamp(0, self._depth_clip) / self._depth_clip

    # Roll buffer
    if self._depth_buffer is None:
      self._depth_buffer = depth.unsqueeze(1).repeat(1, self._enc_seq_len, 1, 1, 1)
    else:
      self._depth_buffer = torch.cat(
        [self._depth_buffer[:, 1:], depth.unsqueeze(1)], dim=1
      )

    self.z_enc = self.depth_encoder(self._depth_buffer)

  def _reset_depth_buffer(self, env_ids: torch.Tensor) -> None:
    """Clear depth buffer for reset environments."""
    if self._depth_buffer is not None:
      self._depth_buffer[env_ids] = 0.0

  # ------------------------------------------------------------------
  # Override step / reset to inject z_enc computation
  # ------------------------------------------------------------------

  def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
    # Run the full base step (physics, terminations, rewards, resets,
    # forward, commands, events, sense, observations).
    result = super().step(action)
    # z_enc is now computed — observations that read env.z_enc will
    # pick it up on the *next* step's observation_manager.compute().
    # For the current step, we recompute observations so z_enc is fresh.
    self._update_z_enc()
    self.obs_buf = self.observation_manager.compute(update_history=False)
    return (self.obs_buf, result[1], result[2], result[3], result[4])

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[types.VecEnvObs, dict]:
    result = super().reset(seed=seed, env_ids=env_ids, options=options)
    if env_ids is not None:
      self._reset_depth_buffer(env_ids)
    else:
      self._depth_buffer = None
    self._update_z_enc()
    self.obs_buf = self.observation_manager.compute(update_history=False)
    return self.obs_buf, result[1]

  # ------------------------------------------------------------------
  # Visualizers
  # ------------------------------------------------------------------

  def update_visualizers(self, vis) -> None:
    super().update_visualizers(vis)
    draw_camera_ball_overlay(self, vis)


@dataclass(kw_only=True)
class DribblingEnvCfg(ManagerBasedRlEnvCfg):
  class_type: ClassVar[type] = DribblingEnv

  # Depth encoder config
  z_enc_dim: int = 64
  depth_enc_height: int = 72
  depth_enc_width: int = 128
  depth_enc_seq_len: int = 5
  depth_clip: float = 6.0
