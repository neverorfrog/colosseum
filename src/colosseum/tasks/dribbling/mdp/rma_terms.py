"""Concrete RMA encoder terms for the dribbling task.

BallRmaTerm pairs two encoders for the ball latent slot:
  privileged_encoder  — GT ball obs (pos, vel) -> MLP -> latent  (Phase 1)
  adaptation_encoder  — depth frame + recurrent hidden state -> latent (Phase 2)

The adaptation path uses:
  - shared DepthEncoder (CNN + GRU) producing a 64D latent
  - BallHead producing [x, y, vx, vy] for supervised Phase 2 loss only

The actor always consumes the shared latent, not BallHead predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from colosseum.algorithm.encoders import BallHead, DepthEncoder, PrivilegedEncoder
from colosseum.managers.rma_manager import RmaTerm, RmaTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BallRmaTermCfg(RmaTermCfg):
  """Config for the dribbling ball encoder term."""

  privileged_obs_group: str = "privileged_ball"
  adaptation_obs_group: str | None = None

  # Shared latent dimensions
  latent_dim: int = 64
  gru_hidden: int = 64

  # Depth preprocessing
  sensor_name: str = "head_rgbd"
  height: int = 60
  width: int = 80
  depth_clip: float = 6.0

  # Training losses
  lambda_pos: float = 1.0
  lambda_vel: float = 0.1
  normalize_targets: bool = True
  tbptt_chunk_len: int = 16
  warmup_steps: int = 4

  # FOV tracking (camera-space projection)
  camera_name: str = "robot/d455_color"
  camera_fovy: float = 60.0
  camera_aspect_ratio: float = 4.0 / 3.0

  def build(self, env: ManagerBasedRlEnv) -> BallRmaTerm:
    return BallRmaTerm(cfg=self, env=env)


class BallRmaTerm(RmaTerm):
  """Privileged MLP + shared depth encoder for dribbling ball information."""

  def __init__(self, cfg: BallRmaTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)

    group_dim = env.observation_manager.group_obs_dim[cfg.privileged_obs_group]
    input_dim = group_dim[0] if isinstance(group_dim, tuple) else int(group_dim)
    self._priv_encoder = PrivilegedEncoder(
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
    ).to(env.device)

    self._depth_encoder = DepthEncoder(
      latent_dim=cfg.latent_dim,
      gru_hidden=cfg.gru_hidden,
    ).to(env.device)
    self._ball_head = BallHead(latent_dim=cfg.latent_dim).to(env.device)

    N = env.num_envs
    device = env.device

    self._gru_hidden = torch.zeros((1, N, cfg.gru_hidden), device=device)
    self._current_frame = torch.zeros((N, 1, cfg.height, cfg.width), device=device)

    self._ball_in_fov = torch.zeros(N, device=device, dtype=torch.bool)
    self._steps_since_reset = torch.zeros(N, device=device, dtype=torch.long)

    self._reset_event = torch.zeros(N, device=device, dtype=torch.bool)
    self._reset_pending = torch.zeros(N, device=device, dtype=torch.bool)

    self._target_mean = torch.zeros(4, device=device)
    self._target_var = torch.ones(4, device=device)

  # ------------------------------------------------------------------
  # Encoder properties
  # ------------------------------------------------------------------

  @property
  def privileged_encoder(self) -> nn.Module:
    return self._priv_encoder

  @property
  def adaptation_encoder(self) -> nn.Module:
    # Return both modules so adaptation_parameters() includes both parameter sets.
    return nn.ModuleList([self._depth_encoder, self._ball_head])

  # ------------------------------------------------------------------
  # Encoding
  # ------------------------------------------------------------------

  def encode_privileged(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return self._priv_encoder(obs_dict[self.cfg.privileged_obs_group])

  def encode_adaptation(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]

    if cfg.adaptation_obs_group is None or cfg.adaptation_obs_group not in obs_dict:
      return torch.zeros(self._env.num_envs, cfg.latent_dim, device=self._env.device)

    frame = obs_dict[cfg.adaptation_obs_group]
    if frame.dim() == 5:
      frame = frame[:, -1]

    z_t, new_hidden = self._depth_encoder(frame, self._gru_hidden)
    self._gru_hidden = new_hidden
    return z_t

  # ------------------------------------------------------------------
  # Adaptation mask and reset events
  # ------------------------------------------------------------------

  def get_adaptation_mask(self) -> torch.Tensor | None:
    return self._ball_in_fov

  def get_reset_event(self) -> torch.Tensor | None:
    return self._reset_event

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def update(self) -> None:
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]

    # reset() marks pending events; update() publishes per-step event mask.
    self._reset_event = self._reset_pending.clone()
    self._reset_pending.zero_()

    try:
      sensor = self._env.scene[cfg.sensor_name]
    except KeyError:
      self._ball_in_fov.fill_(False)
      self._current_frame.zero_()
      return

    depth = sensor.data.depth  # (N, H, W, 1)
    depth = depth.permute(0, 3, 1, 2).float()  # (N, 1, H, W)
    depth = F.interpolate(
      depth,
      size=(cfg.height, cfg.width),
      mode="bilinear",
      align_corners=False,
    )
    depth = depth.clamp(0.0, cfg.depth_clip) / cfg.depth_clip
    self._current_frame = depth

    cam_id = self._env.sim.mj_model.camera(cfg.camera_name).id
    cam_pos = self._env.sim.data.cam_xpos[:, cam_id, :]  # (N, 3)
    cam_mat = self._env.sim.data.cam_xmat[:, cam_id, :].reshape(-1, 3, 3)  # (N, 3, 3)

    ball_pos_w = self._env.scene["ball"].data.root_link_pos_w  # (N, 3)
    p_rel = ball_pos_w - cam_pos
    p_cam = torch.bmm(cam_mat.transpose(1, 2), p_rel.unsqueeze(-1)).squeeze(-1)

    in_front = p_cam[:, 2] < 0
    ball_depth = (-p_cam[:, 2]).clamp_min(1e-6)

    tan_half_v = math.tan(math.radians(cfg.camera_fovy / 2))
    tan_half_h = tan_half_v * cfg.camera_aspect_ratio
    nx = p_cam[:, 0] / (ball_depth * tan_half_h)
    ny = p_cam[:, 1] / (ball_depth * tan_half_v)

    new_in_fov = (
      in_front
      & (nx.abs() <= 1.0)
      & (ny.abs() <= 1.0)
      & (ball_depth < cfg.depth_clip)
    )

    just_entered = new_in_fov & ~self._ball_in_fov
    if just_entered.any():
      self._gru_hidden[:, just_entered, :] = 0.0

    self._reset_event = self._reset_event | just_entered
    self._steps_since_reset = torch.where(
      self._reset_event,
      torch.zeros_like(self._steps_since_reset),
      self._steps_since_reset + 1,
    )
    self._ball_in_fov = new_in_fov

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      return
    self._gru_hidden[:, env_ids, :] = 0.0
    self._current_frame[env_ids] = 0.0
    self._ball_in_fov[env_ids] = False
    self._steps_since_reset[env_ids] = 0
    self._reset_pending[env_ids] = True

  def get_current_adaptation_obs(self) -> dict[str, torch.Tensor]:
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]
    if cfg.adaptation_obs_group is None:
      return {}
    return {cfg.adaptation_obs_group: self._current_frame}

  # ------------------------------------------------------------------
  # Checkpointing (target normalization stats)
  # ------------------------------------------------------------------

  def extra_state_dict(self) -> dict:
    return {
      "target_mean": self._target_mean.cpu(),
      "target_var": self._target_var.cpu(),
    }

  def load_extra_state_dict(self, state: dict) -> None:
    if "target_mean" in state:
      self._target_mean = state["target_mean"].to(self._env.device)
    if "target_var" in state:
      self._target_var = state["target_var"].to(self._env.device)

  # ------------------------------------------------------------------
  # Inference-time ball prediction (for verification / visualization)
  # ------------------------------------------------------------------

  def predict_ball_state(self) -> torch.Tensor | None:
    """Run ball head on current depth frame to get [x, y, vx, vy] prediction.

    Performs a fresh CNN+GRU forward pass on _current_frame using _gru_hidden
    (without mutating it) so the prediction always reflects the latest frame.

    If target normalization was used during training, the output is
    denormalized back to physical units.

    Returns:
      (N, 4) tensor of [x, y, vx, vy] in the same frame as privileged_ball,
      or None if Phase 1 (no adaptation obs group).
    """
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]
    if cfg.adaptation_obs_group is None:
      return None
    with torch.no_grad():
      # Fresh forward pass (does not mutate _gru_hidden)
      z_t, _ = self._depth_encoder(self._current_frame, self._gru_hidden)
      pred = self._ball_head(z_t)  # (N, 4) — possibly normalized
      if cfg.normalize_targets:
        pred = pred * torch.sqrt(self._target_var + 1e-6) + self._target_mean
      return pred

  # ------------------------------------------------------------------
  # Custom adaptation loss
  # ------------------------------------------------------------------

  def compute_loss(
    self,
    privileged_obs: dict[str, torch.Tensor],
    adaptation_obs: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor] | None:
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]

    if cfg.adaptation_obs_group is None:
      return None
    if cfg.adaptation_obs_group not in adaptation_obs:
      return None
    if cfg.privileged_obs_group not in privileged_obs:
      return None

    frames = adaptation_obs[cfg.adaptation_obs_group]
    gt = privileged_obs[cfg.privileged_obs_group]

    if frames.dim() == 4:
      frames = frames.unsqueeze(1)
    if gt.dim() == 2:
      gt = gt.unsqueeze(1)

    B, T = frames.shape[0], frames.shape[1]

    reset_mask = adaptation_obs.get("reset_mask")
    if reset_mask is None:
      reset_mask = torch.zeros((B, T), device=frames.device, dtype=torch.bool)
    else:
      if reset_mask.dim() == 1:
        reset_mask = reset_mask.reshape(B, T)
      reset_mask = reset_mask.to(dtype=torch.bool)

    loss_mask = mask
    if loss_mask is None:
      loss_mask = adaptation_obs.get("loss_mask")
    if loss_mask is None:
      loss_mask = torch.ones((B, T), device=frames.device, dtype=torch.bool)
    else:
      if loss_mask.dim() == 1:
        loss_mask = loss_mask.reshape(B, T)
      loss_mask = loss_mask.to(dtype=torch.bool)

    z_seq, _ = self._depth_encoder.encode_sequence(
      frames,
      hidden=None,
      reset_mask=reset_mask,
      tbptt_chunk_len=cfg.tbptt_chunk_len,
    )
    pred = self._ball_head(z_seq)  # (B, T, 4)

    target = gt
    if cfg.normalize_targets:
      with torch.no_grad():
        flat = gt.reshape(-1, 4)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        momentum = 0.01
        self._target_mean = (1.0 - momentum) * self._target_mean + momentum * batch_mean
        self._target_var = (1.0 - momentum) * self._target_var + momentum * batch_var
      target = (gt - self._target_mean) / torch.sqrt(self._target_var + 1e-6)

    pos_err = (pred[:, :, :2] - target[:, :, :2]).pow(2).mean(dim=-1)
    vel_err = (pred[:, :, 2:] - target[:, :, 2:]).pow(2).mean(dim=-1)

    if loss_mask.any():
      pos_loss = pos_err[loss_mask].mean()
      vel_loss = vel_err[loss_mask].mean()
    else:
      pos_loss = pos_err.sum() * 0.0
      vel_loss = vel_err.sum() * 0.0

    return {
      "ball_pos": cfg.lambda_pos * pos_loss,
      "ball_vel": cfg.lambda_vel * vel_loss,
    }
