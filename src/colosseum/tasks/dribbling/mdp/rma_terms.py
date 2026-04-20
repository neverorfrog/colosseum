"""Unified RMA encoder term for the dribbling task.

DribblingRmaTerm follows the standard RMA two-phase paradigm:
  - Both phases produce the same latent z (latent_dim) that the actor consumes.
  - Phase 1: privileged encoder (MLP) reads all GT info → z.
  - Phase 2: depth encoder (CNN + GRU) reads depth frames → z.
  - The actor is frozen in Phase 2; only the depth encoder is trained.

Phase 1 (privileged, GT inputs):
    priv_encoder:  cat([ball_pos_vel (4D), nearest_obs_norm (4D)])  →  z  (latent_dim)
    nearest_obs:   body-frame pos+vel of the nearest active obstacle, normalised
                   by pos_scale / vel_scale and clipped to ±2.

Phase 2 (visual adaptation):
    DepthEncoder (CNN + GRU):  depth frame  →  z  (latent_dim)
    BallHead:     z  →  [x, y, vx, vy]             (supervision only, not actor input)
    ObstacleHead: z  →  [x,y,vx,vy]                (supervision only, not actor input)

Training losses (Phase 2):
    latent_mse:   ||z_depth - z_priv||²                      (main alignment signal)
    ball_pos:     BallHead[:2]  vs  GT ball position          (auxiliary)
    ball_vel:     BallHead[2:]  vs  GT ball velocity          (auxiliary)
    obstacle_pos: ObstacleHead[:2]   vs  GT nearest obstacle pos     (auxiliary)
    obstacle_vel: ObstacleHead[2:]   vs  GT nearest obstacle vel     (auxiliary)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from colosseum.algorithm.encoders import (
  BallHead,
  DepthEncoder,
  ObstacleHead,
  PrivilegedEncoder,
)
from colosseum.managers.rma_manager import RmaTerm, RmaTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DribblingRmaTermCfg(RmaTermCfg):
  """Configuration for the unified dribbling RMA encoder term."""

  # Privileged observation group names (must match observation_cfg.py).
  privileged_obs_group: str = "privileged_ball"
  obstacle_privileged_obs_group: str = "privileged_obstacles"

  # Adaptation (depth) observation group — None until Phase 2.
  adaptation_obs_group: str | None = None

  # Single latent dimension shared by both phases.
  latent_dim: int = 64

  # DepthEncoder / GRU settings.
  gru_hidden: int = 256

  # Depth frame settings.
  sensor_name: str = "head_rgbd"
  height: int = 108
  width: int = 192
  depth_clip: float = 6.0

  # Obstacle normalisation: divide raw body-frame values before concatenating.
  # Clipped to ±2 after division (1000 m parked obstacles → 2.0).
  obs_pos_scale: float = 5.0   # metres
  obs_vel_scale: float = 1.0   # m/s

  # Training loss weights.
  lambda_ball_pos: float = 1.0
  lambda_ball_vel: float = 0.5
  lambda_obstacle_pos: float = 1.0
  lambda_obstacle_vel: float = 0.5

  # TBPTT / warmup.
  tbptt_chunk_len: int = 16
  warmup_steps: int = 4

  # Ball FOV tracking (for adaptation mask).
  camera_name: str = "robot/d455_color"
  camera_fovy: float = 60.0
  camera_aspect_ratio: float = 4.0 / 3.0

  def build(self, env: ManagerBasedRlEnv) -> DribblingRmaTerm:
    return DribblingRmaTerm(cfg=self, env=env)


# ---------------------------------------------------------------------------
# Term
# ---------------------------------------------------------------------------


class DribblingRmaTerm(RmaTerm):
  """Unified ball + obstacle RMA encoder for dribbling.

  See module docstring for architecture overview.
  """

  cfg: DribblingRmaTermCfg

  def __init__(self, cfg: DribblingRmaTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    N = env.num_envs
    device = env.device
    K = 1

    # ------------------------------------------------------------------
    # Phase 1: single privileged encoder over all GT info.
    # Input: cat([ball_pos_vel (4), obs_norm (4)]) in the current task setup.
    # ------------------------------------------------------------------
    self._priv_encoder = PrivilegedEncoder(
      input_dim=4 + K * 4,
      latent_dim=cfg.latent_dim,
    ).to(device)

    # ------------------------------------------------------------------
    # Phase 2: shared depth encoder + supervision heads.
    # Both heads are auxiliary — their outputs are NOT the actor latent.
    # ------------------------------------------------------------------
    self._depth_encoder = DepthEncoder(
      latent_dim=cfg.latent_dim,
      gru_hidden=cfg.gru_hidden,
    ).to(device)

    self._ball_head = BallHead(latent_dim=cfg.latent_dim).to(device)

    self._obstacle_head = ObstacleHead(latent_dim=cfg.latent_dim).to(device)

    # ------------------------------------------------------------------
    # Runtime state
    # ------------------------------------------------------------------
    self._gru_hidden = torch.zeros((1, N, cfg.gru_hidden), device=device)
    self._current_frame = torch.zeros((N, 1, cfg.height, cfg.width), device=device)

    self._ball_in_fov = torch.zeros(N, device=device, dtype=torch.bool)
    self._steps_since_reset = torch.zeros(N, device=device, dtype=torch.long)
    self._reset_event = torch.zeros(N, device=device, dtype=torch.bool)
    self._reset_pending = torch.zeros(N, device=device, dtype=torch.bool)

    # Ball normalisation constants (used in BallHead loss target).
    self._ball_mean = torch.tensor([0.5, 0.0, 0.0, 0.0], device=device)
    self._ball_std = torch.tensor([1.0, 0.5, 1.5, 1.5], device=device)

    # Cached outputs for visualisation / diagnostics.
    self._last_ball_pred: torch.Tensor | None = None
    self._last_z_adapt: torch.Tensor | None = None

  # ------------------------------------------------------------------
  # Encoder properties
  # ------------------------------------------------------------------

  @property
  def privileged_group_names(self) -> list[str]:
    return [self.cfg.privileged_obs_group, self.cfg.obstacle_privileged_obs_group]

  @property
  def privileged_encoder(self) -> nn.Module:
    return self._priv_encoder

  @property
  def adaptation_encoder(self) -> nn.Module:
    return nn.ModuleList([self._depth_encoder, self._ball_head, self._obstacle_head])

  # ------------------------------------------------------------------
  # Encoding
  # ------------------------------------------------------------------

  def encode_privileged(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    cfg: DribblingRmaTermCfg = self.cfg  # type: ignore[assignment]

    ball_raw = obs_dict[cfg.privileged_obs_group]                  # (N, 4)
    obs_raw = obs_dict[cfg.obstacle_privileged_obs_group]          # (N, K*4)
    obs_norm = self._normalise_obs(obs_raw)                        # (N, K*4)

    priv_input = torch.cat([ball_raw, obs_norm], dim=-1)           # (N, 4 + K*4)
    return self._priv_encoder(priv_input)                          # (N, latent_dim)

  def encode_adaptation(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    cfg: DribblingRmaTermCfg = self.cfg  # type: ignore[assignment]
    N = self._env.num_envs
    device = self._env.device

    if cfg.adaptation_obs_group is None or cfg.adaptation_obs_group not in obs_dict:
      return torch.zeros(N, cfg.latent_dim, device=device)

    frame = obs_dict[cfg.adaptation_obs_group]
    if frame.dim() == 5:
      frame = frame[:, -1]

    z_t, new_hidden = self._depth_encoder(frame, self._gru_hidden)
    self._gru_hidden = new_hidden
    self._last_z_adapt = z_t.detach()

    # Heads are supervision-only — cached for diagnostics, not returned.
    with torch.no_grad():
      pred = self._ball_head(z_t) * self._ball_std + self._ball_mean
      self._last_ball_pred = pred

    return z_t  # (N, latent_dim)

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
    cfg: DribblingRmaTermCfg = self.cfg  # type: ignore[assignment]

    self._reset_event = self._reset_pending.clone()
    self._reset_pending.zero_()

    try:
      sensor = self._env.scene[cfg.sensor_name]
    except KeyError:
      self._ball_in_fov.fill_(False)
      self._current_frame.zero_()
      return

    depth = sensor.data.depth  # (N, H, W, 1)
    depth = depth.permute(0, 3, 1, 2).float()
    depth = F.interpolate(
      depth,
      size=(cfg.height, cfg.width),
      mode="bilinear",
      align_corners=False,
    )
    depth = depth.clamp(0.0, cfg.depth_clip) / cfg.depth_clip
    self._current_frame = depth

    cam_id = self._env.sim.mj_model.camera(cfg.camera_name).id
    cam_pos = self._env.sim.data.cam_xpos[:, cam_id, :]
    cam_mat = self._env.sim.data.cam_xmat[:, cam_id, :].reshape(-1, 3, 3)

    ball_pos_w = self._env.scene["ball"].data.root_link_pos_w
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
    cfg: DribblingRmaTermCfg = self.cfg  # type: ignore[assignment]
    if cfg.adaptation_obs_group is None:
      return {}
    return {cfg.adaptation_obs_group: self._current_frame}

  # ------------------------------------------------------------------
  # Adaptation loss (Phase 2)
  # ------------------------------------------------------------------

  def compute_loss(
    self,
    privileged_obs: dict[str, torch.Tensor],
    adaptation_obs: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor] | None:
    cfg: DribblingRmaTermCfg = self.cfg  # type: ignore[assignment]
    K = 1

    if cfg.adaptation_obs_group is None or cfg.adaptation_obs_group not in adaptation_obs:
      return None
    if cfg.privileged_obs_group not in privileged_obs:
      return None

    frames = adaptation_obs[cfg.adaptation_obs_group]   # (B, T, 1, H, W)
    gt_ball = privileged_obs[cfg.privileged_obs_group]  # (B, T, 4)
    gt_obs = privileged_obs.get(cfg.obstacle_privileged_obs_group)  # (B, T, K*4) or None

    if frames.dim() == 4:
      frames = frames.unsqueeze(1)
    if gt_ball.dim() == 2:
      gt_ball = gt_ball.unsqueeze(1)

    B, T = frames.shape[0], frames.shape[1]

    reset_mask = adaptation_obs.get("reset_mask")
    if reset_mask is None:
      reset_mask = torch.zeros((B, T), device=frames.device, dtype=torch.bool)
    else:
      reset_mask = reset_mask.reshape(B, T).to(dtype=torch.bool)

    loss_mask = mask
    if loss_mask is None:
      loss_mask = adaptation_obs.get("loss_mask")
    if loss_mask is None:
      loss_mask = torch.ones((B, T), device=frames.device, dtype=torch.bool)
    else:
      loss_mask = loss_mask.reshape(B, T).to(dtype=torch.bool)

    # Encode full sequence with depth encoder.
    z_seq, _ = self._depth_encoder.encode_sequence(
      frames,
      hidden=None,
      reset_mask=reset_mask,
      tbptt_chunk_len=cfg.tbptt_chunk_len,
    )  # (B, T, latent_dim)

    # ------------------------------------------------------------------
    # Latent alignment: push z_depth toward z_priv (main RMA signal).
    # ------------------------------------------------------------------
    with torch.no_grad():
      gt_ball_flat = gt_ball.reshape(-1, 4)
      if gt_obs is not None:
        if gt_obs.dim() == 2:
          gt_obs = gt_obs.unsqueeze(1)
        gt_obs_norm = self._normalise_obs(gt_obs.reshape(-1, K * 4)).reshape(B, T, K * 4)
        priv_input = torch.cat([gt_ball_flat, gt_obs_norm.reshape(-1, K * 4)], dim=-1)
      else:
        obs_zeros = torch.zeros(B * T, K * 4, device=frames.device)
        priv_input = torch.cat([gt_ball_flat, obs_zeros], dim=-1)
      z_priv = self._priv_encoder(priv_input).reshape(B, T, -1)
    latent_err = (z_seq - z_priv).pow(2).mean(dim=-1)

    # ------------------------------------------------------------------
    # Ball supervision (BallHead auxiliary).
    # ------------------------------------------------------------------
    ball_pred = self._ball_head(z_seq)  # (B, T, 4)
    target_ball = (gt_ball - self._ball_mean) / self._ball_std
    pos_err = (ball_pred[:, :, :2] - target_ball[:, :, :2]).pow(2).mean(dim=-1)
    vel_err = (ball_pred[:, :, 2:] - target_ball[:, :, 2:]).pow(2).mean(dim=-1)

    # ------------------------------------------------------------------
    # Obstacle supervision (ObstacleHead auxiliary).
    # ------------------------------------------------------------------
    obs_losses: dict[str, torch.Tensor] = {}
    if gt_obs is not None:
      obs_pred = self._obstacle_head(z_seq)   # (B, T, K*4)
      obs_pos_err = (obs_pred[:, :, :K*2] - gt_obs_norm[:, :, :K*2]).pow(2).mean(dim=-1)
      obs_vel_err = (obs_pred[:, :, K*2:] - gt_obs_norm[:, :, K*2:]).pow(2).mean(dim=-1)
      if loss_mask.any():
        obs_losses["obstacle_pos"] = cfg.lambda_obstacle_pos * obs_pos_err[loss_mask].mean()
        obs_losses["obstacle_vel"] = cfg.lambda_obstacle_vel * obs_vel_err[loss_mask].mean()
      else:
        obs_losses["obstacle_pos"] = obs_pos_err.sum() * 0.0
        obs_losses["obstacle_vel"] = obs_vel_err.sum() * 0.0

    if loss_mask.any():
      latent_loss = latent_err[loss_mask].mean()
      pos_loss = pos_err[loss_mask].mean()
      vel_loss = vel_err[loss_mask].mean()
    else:
      latent_loss = latent_err.sum() * 0.0
      pos_loss = pos_err.sum() * 0.0
      vel_loss = vel_err.sum() * 0.0

    return {
      "latent_mse": latent_loss,
      "ball_pos": cfg.lambda_ball_pos * pos_loss,
      "ball_vel": cfg.lambda_ball_vel * vel_loss,
      **obs_losses,
    }

  # ------------------------------------------------------------------
  # Checkpointing
  # ------------------------------------------------------------------

  def extra_state_dict(self) -> dict:
    return {}

  def load_extra_state_dict(self, state: dict) -> None:
    del state

  # ------------------------------------------------------------------
  # Visualisation helpers
  # ------------------------------------------------------------------

  def predict_ball_state(self) -> torch.Tensor | None:
    """Latest ball head prediction [x, y, vx, vy] cached from encode_adaptation."""
    return self._last_ball_pred

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _normalise_obs(self, obs_raw: torch.Tensor) -> torch.Tensor:
    """Normalise raw body-frame obstacle coordinates and clip to ±2.

    Layout: [..., x, y, vx, vy]
    Positions divided by obs_pos_scale, velocities by obs_vel_scale.
    Inactive obstacles at 1000 m clip to 2.0 — a clear "far away" signal.
    """
    cfg: DribblingRmaTermCfg = self.cfg  # type: ignore[assignment]
    K = 1
    pos = obs_raw[..., : K * 2] / cfg.obs_pos_scale
    vel = obs_raw[..., K * 2 :] / cfg.obs_vel_scale
    return torch.cat([pos, vel], dim=-1).clamp(-2.0, 2.0)
