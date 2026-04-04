"""Depth encoder for visual ball state estimation.

Architecture from the dribbling paper:
  - Per-frame CNN: 1×64×64 depth → 128D embedding
  - LSTM: 5 embeddings → 256D hidden state
  - Linear head: 256 → 64D latent (z_enc)

The projection head predicts GT (nx, ny) for auxiliary supervision
during training and is discarded at deployment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DepthEncoder(nn.Module):
  """CNN + LSTM depth encoder → z_enc ∈ ℝ⁶⁴.

  Input:  (B, T, 1, H, W) — T depth frames
  Output: (B, latent_dim)  — z_enc latent

  The CNN uses AdaptiveAvgPool2d so any spatial resolution works,
  but the convolution layers assume at least ~16×16 input.
  """

  def __init__(
    self,
    latent_dim: int = 64,
    input_height: int = 72,
    input_width: int = 128,
  ) -> None:
    super().__init__()
    self.latent_dim = latent_dim
    self.input_height = input_height
    self.input_width = input_width

    # Per-frame CNN: (1, 64, 64) → 128D
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, 5, 2),  # → (32, 30, 30)
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, 2),  # → (64, 14, 14)
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 128, 3, 2),  # → (128, 6, 6)
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d(1),  # → (128, 1, 1)
      nn.Flatten(),  # → (128,)
      nn.Linear(128, 128),
    )

    # Temporal aggregation: sequence of 128D embeddings → 256D hidden
    self.lstm = nn.LSTM(128, 256, batch_first=True)

    # Latent projection: 256D → latent_dim
    self.head = nn.Linear(256, latent_dim)

  def forward(self, frames: Tensor) -> Tensor:
    """Encode a sequence of depth frames.

    Args:
      frames: (B, T, 1, H, W) depth frames, normalized to [0, 1].

    Returns:
      z_enc: (B, latent_dim) latent vector.
    """
    B, T, C, H, W = frames.shape
    e = self.cnn(frames.reshape(B * T, C, H, W))  # (B*T, 128)
    e = e.reshape(B, T, -1)  # (B, T, 128)
    _, (h, _) = self.lstm(e)  # h: (1, B, 256)
    return self.head(h.squeeze(0))  # (B, latent_dim)


class ProjectionHead(nn.Module):
  """Auxiliary head predicting GT (nx, ny) from z_enc.

  Used only during training for supervision. Discarded at deployment.
  """

  def __init__(self, latent_dim: int = 64) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(latent_dim, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, 2),
    )

  def forward(self, z_enc: Tensor) -> Tensor:
    """Predict normalized image coordinates (nx, ny).

    Args:
      z_enc: (B, latent_dim) encoder output.

    Returns:
      nxny: (B, 2) predicted (nx, ny) in [-1, 1]².
    """
    return self.net(z_enc)
