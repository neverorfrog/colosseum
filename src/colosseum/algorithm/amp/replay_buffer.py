"""Circular buffer of POLICY AMP transitions ``(state, next_state)``.

Distinct from the PPO rollout buffer in three ways: it holds only AMP-obs pairs
(not full transitions), it feeds the discriminator (not the policy gradient), and
it persists across iterations (large, ~100k) so the discriminator sees the policy's
transitions mixed over many rollouts — which stabilizes adversarial training.
"""

from __future__ import annotations

from typing import Iterator

import torch


class AMPReplayBuffer:
  def __init__(self, obs_dim: int, buffer_size: int, device: torch.device | str) -> None:
    self.device = torch.device(device)
    self.states = torch.zeros(buffer_size, obs_dim, device=self.device)
    self.next_states = torch.zeros(buffer_size, obs_dim, device=self.device)
    self.buffer_size = buffer_size
    self.step = 0
    self.num_samples = 0

  def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
    """Append a batch of transitions, wrapping around when full."""
    n = states.shape[0]
    start = self.step
    end = self.step + n
    if end > self.buffer_size:
      first = self.buffer_size - self.step
      self.states[start:] = states[:first]
      self.next_states[start:] = next_states[:first]
      self.states[: end - self.buffer_size] = states[first:]
      self.next_states[: end - self.buffer_size] = next_states[first:]
    else:
      self.states[start:end] = states
      self.next_states[start:end] = next_states
    self.num_samples = min(self.buffer_size, max(end, self.num_samples))
    self.step = end % self.buffer_size

  def feed_forward_generator(
    self, num_mini_batches: int, mini_batch_size: int
  ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for _ in range(num_mini_batches):
      idx = torch.randint(0, self.num_samples, (mini_batch_size,), device=self.device)
      yield self.states[idx], self.next_states[idx]
