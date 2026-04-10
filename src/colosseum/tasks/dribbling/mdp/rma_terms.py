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

update() rolls the depth buffer each step and tracks whether the ball is within
the camera's horizontal field of view using GT state (available during training).

When ball is in FOV and the buffer is warmed up (>= seq_len frames), the term
reports itself as having valid adaptation signal via get_adaptation_mask(). The
RmaManager uses this mask to:
  - Blend depth/privileged encoders during Phase 2 collection (fallback to
    privileged when ball is out of FOV, so the policy stays well-conditioned).
  - Compute adaptation loss only on valid samples (temporally-aligned mask
    snapshotted at each collection step).

The depth buffer is zeroed on FOV re-entry so the LSTM always starts from a
clean slate rather than processing stale out-of-FOV frames.
"""

from __future__ import annotations

import math
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

  # --- FOV tracking (camera-space projection) ---
  camera_name: str = "robot/d455_color"
  """MuJoCo camera name used for the ball-in-FOV projection check.
  Must match the camera defined in the robot XML and used by head_ball_tracking."""

  camera_fovy: float = 60.0
  """Vertical field of view in degrees (MuJoCo fovy parameter).
  RealSense D455 default in the T1 XML."""

  camera_aspect_ratio: float = 4.0 / 3.0
  """Width / height pixel ratio of the depth sensor output.
  Used to derive the horizontal FOV from fovy."""

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
    self._ball_in_fov: torch.Tensor | None = None       # (N,) bool
    self._frames_since_entry: torch.Tensor | None = None  # (N,) int, capped at seq_len

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

    Pure depth encoder — no fallback logic here. The fallback to the privileged
    encoder for out-of-FOV envs is handled by RmaManager.encode_phase2_with_fallback()
    during collection. At learning time this always runs the depth encoder so
    that gradients flow correctly through the snapshotted depth frames.
    """
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]

    if cfg.adaptation_obs_group is not None and cfg.adaptation_obs_group in obs_dict:
      return self._depth_encoder(obs_dict[cfg.adaptation_obs_group])

    return torch.zeros(self._env.num_envs, cfg.latent_dim, device=self._env.device)

  # ------------------------------------------------------------------
  # Adaptation mask
  # ------------------------------------------------------------------

  def get_adaptation_mask(self) -> torch.Tensor | None:
    """Return (N,) bool mask of envs with a valid adaptation signal.

    Valid = ball is currently within the camera's horizontal FOV AND the depth
    buffer has been populated with at least seq_len in-FOV frames since the last
    re-entry (buffer warm-up).

    Returns None during Phase 1 (camera absent, update() is a no-op so these
    tensors are never initialised).
    """
    if self._ball_in_fov is None or self._frames_since_entry is None:
      return None
    return self._ball_in_fov & (self._frames_since_entry >= self.cfg.seq_len)

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def update(self) -> None:
    """Read depth sensor, update FOV state, roll depth buffer.

    No-op when the camera sensor is absent from the scene (Phase 1 training).

    On FOV re-entry (out-of-FOV → in-FOV transition) the depth buffer is zeroed
    for the re-entering environments so the LSTM starts from a clean slate.
    """
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]

    try:
      sensor = self._env.scene[cfg.sensor_name]
    except KeyError:
      return  # camera absent (Phase 1 training)

    # --- Depth frame preprocessing ---
    depth = sensor.data.depth  # (N, H, W, 1)
    depth = depth.permute(0, 3, 1, 2).float()  # (N, 1, H, W)
    if depth.shape[-2:] != (cfg.height, cfg.width):
      depth = F.interpolate(
        depth,
        size=(cfg.height, cfg.width),
        mode="bilinear",
        align_corners=False,
      )
    depth = depth.clamp(0.0, cfg.depth_clip) / cfg.depth_clip  # [0, 1]

    # --- Ball-in-FOV check using actual camera projection ---
    # Uses cam_xmat (the real head/camera orientation including neck joints)
    # so the check correctly accounts for head tracking and vertical FOV.
    # Mirrors _project_ball_to_camera() in rewards.py.
    cam_id = self._env.sim.mj_model.camera(cfg.camera_name).id
    cam_pos = self._env.sim.data.cam_xpos[:, cam_id, :]         # (N, 3)
    cam_mat = self._env.sim.data.cam_xmat[:, cam_id, :].reshape(-1, 3, 3)  # (N, 3, 3)

    ball_pos_w = self._env.scene["ball"].data.root_link_pos_w   # (N, 3)
    p_rel = ball_pos_w - cam_pos                                # (N, 3)

    # World → camera frame: p_cam = R^T @ p_rel
    p_cam = torch.bmm(cam_mat.transpose(1, 2), p_rel.unsqueeze(-1)).squeeze(-1)  # (N, 3)

    # MuJoCo camera convention: optical axis = -z, so ball_depth = -p_cam_z
    in_front = p_cam[:, 2] < 0                                    # (N,) bool
    ball_depth = (-p_cam[:, 2]).clamp_min(1e-6)                   # (N,)

    # Perspective divide → normalised image coords (±1 = FOV edge)
    tan_half_v = math.tan(math.radians(cfg.camera_fovy / 2))
    tan_half_h = tan_half_v * cfg.camera_aspect_ratio
    nx = p_cam[:, 0] / (ball_depth * tan_half_h)  # (N,)  right = +1 at edge
    ny = p_cam[:, 1] / (ball_depth * tan_half_v)  # (N,)  up    = +1 at edge

    new_in_fov = (
      in_front
      & (nx.abs() <= 1.0)
      & (ny.abs() <= 1.0)
      & (ball_depth < cfg.depth_clip)
    )

    N = self._env.num_envs
    device = self._env.device

    if self._depth_buffer is None:
      # First call — initialise all state tensors
      self._depth_buffer = (
        depth.unsqueeze(1).expand(-1, cfg.seq_len, -1, -1, -1).clone()
      )  # (N, seq_len, 1, H, W)
      self._ball_in_fov = new_in_fov.clone()
      self._frames_since_entry = new_in_fov.long()
    else:
      # Detect out-of-FOV → in-FOV transitions and zero buffer on re-entry
      just_entered = new_in_fov & ~self._ball_in_fov
      if just_entered.any():
        self._depth_buffer[just_entered] = 0.0
        self._frames_since_entry[just_entered] = 0

      # Roll depth buffer: drop oldest frame, append newest
      self._depth_buffer = torch.cat(
        [self._depth_buffer[:, 1:], depth.unsqueeze(1)], dim=1
      )

      # Increment counter for in-FOV envs (capped at seq_len); reset for out-of-FOV
      self._frames_since_entry = torch.where(
        new_in_fov,
        (self._frames_since_entry + 1).clamp(max=cfg.seq_len),
        torch.zeros(N, device=device, dtype=torch.long),
      )
      self._ball_in_fov = new_in_fov

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    """Zero depth buffer and FOV state for reset environments."""
    if env_ids is None:
      return
    if self._depth_buffer is not None:
      self._depth_buffer[env_ids] = 0.0
    if self._ball_in_fov is not None:
      self._ball_in_fov[env_ids] = False
    if self._frames_since_entry is not None:
      self._frames_since_entry[env_ids] = 0

  def get_current_adaptation_obs(self) -> dict[str, torch.Tensor]:
    """Return current depth buffer keyed by adaptation_obs_group."""
    cfg: BallRmaTermCfg = self.cfg  # type: ignore[assignment]
    if cfg.adaptation_obs_group is None or self._depth_buffer is None:
      return {}
    return {cfg.adaptation_obs_group: self._depth_buffer}
