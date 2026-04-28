from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PrivilegedEncoder(nn.Module):
  """Small MLP encoding GT privileged observations to a latent vector.

  Fixed architecture: input_dim → 32 → latent_dim, ELU activations, LayerNorm output.
  Used in Phase 1. Replaced by an adaptation encoder (e.g. DepthEncoder) in Phase 2.

  Args:
    input_dim:  Dimensionality of the GT privileged input.
    latent_dim: Output latent dimension (should match the replacing encoder).
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
    return self.net(x)
