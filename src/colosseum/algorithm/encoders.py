"""Encoder networks for RMA-style privileged information encoding.

Current architecture:
  PrivilegedEncoder  — small MLP, used in Phase 1 with GT inputs
  DepthEncoder       — shared depth encoder (CNN + GRU), used in Phase 2
  BallHead           — task head predicting [x, y, vx, vy] from DepthEncoder latent
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PrivilegedEncoder(nn.Module):
  """Small MLP encoding GT privileged observations to a latent vector.

  Fixed architecture: input_dim → 32 → latent_dim, ELU activations, LayerNorm output.
  Used in Phase 1. Replaced by DepthEncoder when visual training begins.

  Args:
    input_dim:  Dimensionality of the GT privileged input.
    latent_dim: Output latent dimension (should match the replacing DepthEncoder).
  """

  def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
    super().__init__()
    self.latent_dim = latent_dim  # type: ignore
    self.net = nn.Sequential(
      nn.Linear(input_dim, 32),
      nn.ELU(),
      nn.Linear(32, latent_dim),
      nn.LayerNorm(latent_dim),
    )

  def forward(self, x: Tensor) -> Tensor:
    """Args:
      x: (B, input_dim) GT privileged values.
    Returns:
      z: (B, latent_dim) latent vector.
    """
    return self.net(x)


class DepthEncoder(nn.Module):
  """Shared depth encoder (CNN + GRU) producing a latent for policy input.

  Single-step inference path:
    frame (B, 1, H, W), hidden (1, B, Hh) -> z_t (B, latent_dim), new_hidden

  Sequence training path:
    frames (B, T, 1, H, W), reset_mask (B, T) -> z_seq (B, T, latent_dim)

  Uses standard Conv2d + GroupNorm (no BatchNorm running stats, which are
  a known RL footgun — running stats drift during training and the frozen
  privileged target has no such state to mirror).

  Args:
    latent_dim: Shared latent dimension exposed to the actor.
    gru_hidden: GRU hidden width.
  """

  def __init__(self, latent_dim: int = 64, gru_hidden: int = 256) -> None:
    super().__init__()
    self.latent_dim = latent_dim  # type: ignore[assignment]
    self.gru_hidden = gru_hidden

    # Per-frame CNN: (1, 108, 192) -> latent_dim
    # Spatial downsampling through 4 stride-2 convs:
    #   108 -> 54 -> 27 -> 14 -> 7
    #   192 -> 96 -> 48 -> 24 -> 12
    # Final map (128, 7, 12) = 10752 features, flattened (not pooled) so the
    # linear head can use spatial position — essential for ball localization.
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
      nn.GroupNorm(8, 32),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
      nn.GroupNorm(8, 64),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
      nn.GroupNorm(8, 128),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
      nn.GroupNorm(8, 128),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(128 * 7 * 12, 256),
      nn.ReLU(),
      nn.Linear(256, latent_dim),
    )

    # Temporal aggregation over frame embeddings.
    self.gru = nn.GRU(latent_dim, gru_hidden, batch_first=True)

    # Project recurrent state back to actor latent size.
    self.head = nn.Linear(gru_hidden, latent_dim)

    # Match PrivilegedEncoder's LayerNorm output so z_adapt lives on the same
    # manifold as z_priv. Without this, the encoder's output has unbounded
    # magnitude and the frozen actor sees out-of-distribution latents.
    self.output_norm = nn.LayerNorm(latent_dim)

  def forward(self, frame: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Encode one frame and advance GRU state.

    Args:
      frame:  (B, 1, H, W) depth frame normalized to [0, 1].
      hidden: Optional previous hidden state (1, B, gru_hidden).

    Returns:
      z_t:       (B, latent_dim)
      new_hidden:(1, B, gru_hidden)
    """
    emb = self.cnn(frame)  # (B, latent_dim)
    out, new_hidden = self.gru(emb.unsqueeze(1), hidden)
    z_t = self.head(out[:, 0, :])
    z_t = self.output_norm(z_t)
    return z_t, new_hidden

  def encode_sequence(
    self,
    frames: Tensor,
    hidden: Tensor | None = None,
    reset_mask: Tensor | None = None,
    tbptt_chunk_len: int = 16,
  ) -> tuple[Tensor, Tensor]:
    """Encode a sequence with optional hidden resets and TBPTT chunking.

    Args:
      frames:          (B, T, 1, H, W)
      hidden:          Optional initial hidden (1, B, gru_hidden).
      reset_mask:      Optional bool tensor (B, T). Hidden is zeroed where True.
      tbptt_chunk_len: Detach hidden every N steps to bound BPTT memory.

    Returns:
      z_seq:       (B, T, latent_dim)
      final_hidden:(1, B, gru_hidden)
    """
    B, T, C, H, W = frames.shape
    if hidden is None:
      hidden = frames.new_zeros((1, B, self.gru_hidden))

    z_list: list[Tensor] = []
    for t in range(T):
      if reset_mask is not None:
        reset_t = reset_mask[:, t].to(dtype=torch.bool)
        if reset_t.any():
          hidden = hidden.clone()
          hidden[:, reset_t, :] = 0.0

      z_t, hidden = self.forward(frames[:, t], hidden)
      z_list.append(z_t)

      if tbptt_chunk_len > 0 and (t + 1) % tbptt_chunk_len == 0:
        hidden = hidden.detach()

    z_seq = torch.stack(z_list, dim=1)
    return z_seq, hidden


class BallHead(nn.Module):
  """Task head predicting [x, y, vx, vy] from shared latent."""

  def __init__(self, latent_dim: int = 64) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(latent_dim, 32),
      nn.ReLU(),
      nn.Linear(32, 4),
    )

  def forward(self, z: Tensor) -> Tensor:
    """Predict ball state from latent.

    Args:
      z: (..., latent_dim)

    Returns:
      (..., 4) tensor ordered as [x, y, vx, vy]
    """
    return self.net(z)
