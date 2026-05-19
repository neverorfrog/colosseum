from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PrivilegedEncoder(nn.Module):
  """Small MLP encoding GT privileged observations to a latent vector.

  Architecture: input_dim → hidden_dim → latent_dim, ELU activations, LayerNorm output.
  Used in Phase 1. Replaced by an adaptation encoder (e.g. DepthEncoder) in Phase 2.

  Args:
    input_dim:  Dimensionality of the GT privileged input.
    latent_dim: Output latent dimension (should match the replacing encoder).
    hidden_dim: Width of the single hidden layer (default 32).
  """

  def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 32) -> None:
    super().__init__()
    self.latent_dim = latent_dim  # type: ignore
    self.net = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ELU(),
      nn.Linear(hidden_dim, latent_dim),
      nn.LayerNorm(latent_dim),
    )

  def forward(self, x: Tensor) -> Tensor:
    return self.net(x)
