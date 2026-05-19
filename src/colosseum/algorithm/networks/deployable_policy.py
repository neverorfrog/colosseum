"""Thin nn.Module wrapper for ONNX-clean deployment of an RMA policy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DeployablePolicy(nn.Module):
  """Actor + adaptation encoder packaged for ONNX export.

  Hardwires the adaptation encoder path (no phase flags, no privileged encoder,
  no runtime branching) so the computation graph is static and ONNX-exportable.

  The caller is responsible for:
    - Normalising actor_obs before passing it in.
    - Maintaining the rolling proprio-history buffer (for ProprioWindowEncoder)
      or depth-frame buffer (for DepthEncoder) and passing the result as
      adaptation_obs.

  Args:
    actor:       Trained PpoActor (or any nn.Module returning action means).
    rma_manager: RmaManager with trained adaptation encoders.
  """

  def __init__(self, actor: nn.Module, rma_manager: nn.Module) -> None:
    super().__init__()
    self.actor = actor
    self.rma_manager = rma_manager

  def forward(self, actor_obs: Tensor, adaptation_obs: dict[str, Tensor]) -> Tensor:
    """Run the adaptation encoder then the actor.

    Args:
      actor_obs:      (B, actor_obs_dim) normalised proprioceptive observation.
      adaptation_obs: Dict of sensor observations consumed by the adaptation
                      encoders (keys match each term's adaptation_obs_group).

    Returns:
      action_means: (B, action_dim) deterministic action means.
    """
    z = self.rma_manager.encode(adaptation_obs, use_adaptation=True, apply_noise=False)
    return self.actor(torch.cat([actor_obs, z], dim=-1))
