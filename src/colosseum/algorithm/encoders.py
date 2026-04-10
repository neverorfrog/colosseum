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


class DepthwiseSeparableConv(nn.Module):
  """Depthwise-separable conv block with stride-2 downsampling.

  Layout: depthwise 3x3 (groups=in_ch) -> pointwise 1x1 -> BN -> ReLU.
  """

  def __init__(self, in_ch: int, out_ch: int) -> None:
    super().__init__()
    self.depthwise = nn.Conv2d(
      in_ch,
      in_ch,
      kernel_size=3,
      stride=2,
      padding=1,
      groups=in_ch,
      bias=False,
    )
    self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    self.bn = nn.BatchNorm2d(out_ch)
    self.act = nn.ReLU()

  def forward(self, x: Tensor) -> Tensor:
    x = self.depthwise(x)
    x = self.pointwise(x)
    x = self.bn(x)
    return self.act(x)


class DepthEncoder(nn.Module):
  """Shared depth encoder (CNN + GRU) producing a latent for policy input.

  Single-step inference path:
    frame (B, 1, H, W), hidden (1, B, Hh) -> z_t (B, latent_dim), new_hidden

  Sequence training path:
    frames (B, T, 1, H, W), reset_mask (B, T) -> z_seq (B, T, latent_dim)

  Args:
    latent_dim: Shared latent dimension exposed to the actor.
    gru_hidden: GRU hidden width.
  """

  def __init__(self, latent_dim: int = 64, gru_hidden: int = 64) -> None:
    super().__init__()
    self.latent_dim = latent_dim  # type: ignore[assignment]
    self.gru_hidden = gru_hidden

    # Per-frame CNN: (1, H, W) -> latent_dim
    self.cnn = nn.Sequential(
      DepthwiseSeparableConv(1, 16),
      DepthwiseSeparableConv(16, 32),
      DepthwiseSeparableConv(32, 64),
      DepthwiseSeparableConv(64, 64),
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
    )

    # Temporal aggregation over frame embeddings.
    self.gru = nn.GRU(latent_dim, gru_hidden, batch_first=True)

    # Project recurrent state back to actor latent size.
    self.head = nn.Linear(gru_hidden, latent_dim)

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
