"""Encoder networks for RMA-style privileged information encoding.

Three encoders:
  PrivilegedEncoder  — small MLP, used in Phase 1 with GT inputs
  DepthEncoder       — CNN + LSTM, used in Phase 2 with depth frames
  ProjectionHead     — auxiliary head for (nx, ny) supervision (Phase 2 only)
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
  """CNN + LSTM depth encoder producing a latent matching PrivilegedEncoder.

  Encodes a sequence of depth frames from a head-mounted camera into a compact
  latent that can directly replace the PrivilegedEncoder output (same latent_dim).

  Input:  (B, T, 1, H, W) depth frames, clipped and normalized to [0, 1]
  Output: (B, latent_dim)

  The CNN uses AdaptiveAvgPool2d so any spatial resolution >= 16x16 works.

  Args:
    latent_dim:  Output latent dimension. Must match the PrivilegedEncoder it replaces.
    lstm_hidden: LSTM hidden size (internal temporal representation width).
  """

  def __init__(self, latent_dim: int = 8, lstm_hidden: int = 64, cnn_chunk: int = 256) -> None:
    super().__init__()
    self.latent_dim = latent_dim  # type: ignore
    self.cnn_chunk = cnn_chunk
    """Max frames processed by the CNN in a single call.
    Prevents OOM when mini-batch × seq_len is large.  256 frames × 72×128 costs
    ~130 MB for the first conv feature map — safe on an 11 GB card."""

    # Per-frame CNN: (1, H, W) → 128D
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, 5, 2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, 2),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, 3, 2),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),  # → (128,)
    )

    # Temporal aggregation: sequence of 128D frame embeddings → lstm_hidden
    self.lstm = nn.LSTM(128, lstm_hidden, batch_first=True)

    # Latent projection: lstm_hidden → latent_dim
    self.head = nn.Linear(lstm_hidden, latent_dim)

  def forward(self, frames: Tensor) -> Tensor:
    """Args:
      frames: (B, T, 1, H, W) depth frames, normalized to [0, 1].
    Returns:
      z: (B, latent_dim) latent vector.
    """
    B, T, C, H, W = frames.shape
    flat = frames.reshape(B * T, C, H, W)  # (B*T, 1, H, W)
    if self.cnn_chunk > 0 and flat.shape[0] > self.cnn_chunk:
      e = torch.cat(
        [self.cnn(flat[i : i + self.cnn_chunk]) for i in range(0, flat.shape[0], self.cnn_chunk)],
        dim=0,
      )  # (B*T, 128)
    else:
      e = self.cnn(flat)  # (B*T, 128)
    e = e.reshape(B, T, -1)  # (B, T, 128)
    _, (h, _) = self.lstm(e)  # h: (1, B, lstm_hidden)
    return self.head(h.squeeze(0))  # (B, latent_dim)


class ProjectionHead(nn.Module):
  """Auxiliary head predicting GT (nx, ny) from encoder latent.

  Used only during Phase 2 training for ball projection supervision.
  Discarded at deployment.

  Args:
    latent_dim: Must match the DepthEncoder latent_dim it attaches to.
  """

  def __init__(self, latent_dim: int = 8) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(latent_dim, 32),
      nn.ReLU(),
      nn.Linear(32, 2),
    )

  def forward(self, z: Tensor) -> Tensor:
    """Args:
      z: (B, latent_dim) encoder output.
    Returns:
      nxny: (B, 2) predicted (nx, ny) in [-1, 1]².
    """
    return self.net(z)
