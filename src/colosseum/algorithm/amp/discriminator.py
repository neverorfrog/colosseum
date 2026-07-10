"""AMP discriminator: the "judge" that scores motion transitions.

A small MLP over a concatenated transition ``[s, s']`` (each ``s`` is a 52-dim AMP
observation, so input is 104-dim). Trained — least-squares GAN — to regress expert
transitions to ``+1`` and policy transitions to ``-1``. Its scalar output is turned
into the AMP style reward, then blended with the task reward.

The expert data appears ONLY in the discriminator loss (computed by the algorithm).
This module just (a) scores transitions and (b) maps a policy score to a reward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd


class AMPDiscriminator(nn.Module):
  def __init__(
    self,
    input_dim: int,
    amp_reward_coef: float,
    hidden_layer_sizes: list[int],
    device: torch.device | str,
    task_reward_lerp: float = 0.0,
  ) -> None:
    """
    Args:
      input_dim: ``2 * amp_obs_dim`` (state + next_state concatenated).
      amp_reward_coef: scales the style reward.
      hidden_layer_sizes: discriminator trunk widths, e.g. ``[256, 256]``.
      task_reward_lerp: blend factor ``lambda`` in
        ``(1 - lambda) * r_amp + lambda * r_task``. 1.0 -> task only, 0.0 -> AMP only.
    """
    super().__init__()
    self.device = torch.device(device)
    self.input_dim = input_dim
    self.amp_reward_coef = amp_reward_coef
    self.task_reward_lerp = task_reward_lerp

    layers: list[nn.Module] = []
    curr = input_dim
    for h in hidden_layer_sizes:
      layers.append(nn.Linear(curr, h))
      layers.append(nn.ReLU())
      curr = h
    self.trunk = nn.Sequential(*layers).to(self.device)
    self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(self.device)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.amp_linear(self.trunk(x))

  def compute_grad_pen(
    self,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
    lambda_: float = 10.0,
  ) -> torch.Tensor:
    """Gradient penalty on expert data, keeping the discriminator smooth (stable GAN)."""
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
    expert_data.requires_grad = True
    disc = self.amp_linear(self.trunk(expert_data))
    grad = autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=torch.ones_like(disc),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return lambda_ * grad.norm(2, dim=1).pow(2).mean()

  def predict_amp_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer: nn.Module | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map a POLICY transition to the (blended) reward. No expert data here.

    Returns ``(reward, d, amp_reward)``: the blended per-step reward fed to PPO,
    the raw discriminator score (for logging), and the unblended style reward.
    """
    with torch.no_grad():
      self.eval()
      if normalizer is not None:
        state = normalizer(state)
        next_state = normalizer(next_state)
      d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
      amp_reward = self.amp_reward_coef * torch.clamp(
        1 - 0.25 * torch.square(d - 1), min=0
      )
      reward = amp_reward
      if self.task_reward_lerp > 0:
        reward = (1.0 - self.task_reward_lerp) * reward + self.task_reward_lerp * (
          task_reward.unsqueeze(-1)
        )
      self.train()
    return reward.squeeze(-1), d.squeeze(-1), amp_reward.squeeze(-1)
