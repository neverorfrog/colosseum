"""Fixed-horizon on-policy rollout storage for PPO.

Stores a full rollout of shape [num_steps, num_envs, dim], computes GAE
returns/advantages, and yields shuffled mini-batches for PPO updates.

Design follows RSL-RL's RolloutStorage and holosoma's RolloutStorage patterns:
- Pre-allocated tensors for zero-allocation collection
- GAE computed in-place after rollout completes
- Mini-batch generator flattens [T, N] -> [T*N] and yields random slices
"""

from __future__ import annotations

from typing import Generator

import torch


class RolloutBuffer:
    """Fixed-horizon on-policy rollout storage.

    Shape convention: [num_steps, num_envs, dim]

    Usage:
        1. Call add() for each step during rollout collection
        2. Call compute_returns_and_advantages() with bootstrapped last values
        3. Iterate mini_batch_generator() for PPO updates
        4. Call clear() to reset for next rollout
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: torch.device,
        extras: dict[str, int] | None = None,
        privileged_obs_dims: dict[str, int] | None = None,
        adaptation_obs_dims: dict[str, int] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.step = 0

        # Pre-allocate all tensors [T, N, dim]
        self.actor_obs = torch.zeros(num_steps, num_envs, actor_obs_dim, device=device)
        self.critic_obs = torch.zeros(num_steps, num_envs, critic_obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=device)
        self.dones = torch.zeros(num_steps, num_envs, 1, device=device)
        self.values = torch.zeros(num_steps, num_envs, 1, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, 1, device=device)
        self.action_means = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.action_stds = torch.zeros(num_steps, num_envs, action_dim, device=device)

        # Computed after rollout
        self.returns = torch.zeros(num_steps, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, 1, device=device)

        # Optional extra tensors (e.g., GT directions for auxiliary losses)
        # Maps name → pre-allocated [T, N, dim] tensor
        self._extras: dict[str, torch.Tensor] = {}
        if extras:
            for name, dim in extras.items():
                self._extras[name] = torch.zeros(
                    num_steps, num_envs, dim, device=device
                )

        # Privileged obs storage (RMA): each group stored separately so
        # encoders can re-run with gradients during learning.
        # Maps group_name → [T, N, group_dim] tensor.
        self._privileged_obs: dict[str, torch.Tensor] = {}
        if privileged_obs_dims:
            for group_name, dim in privileged_obs_dims.items():
                self._privileged_obs[group_name] = torch.zeros(
                    num_steps, num_envs, dim, device=device
                )

        # Adaptation obs storage (Phase 3): proprio windows stored flattened as
        # [T, N, W*D] so the mini-batch generator can index them with the same
        # permutation used for all other tensors.
        self._adaptation_obs: dict[str, torch.Tensor] = {}
        if adaptation_obs_dims:
            for group_name, dim in adaptation_obs_dims.items():
                self._adaptation_obs[group_name] = torch.zeros(
                    num_steps, num_envs, dim, device=device
                )

    def add(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        action_means: torch.Tensor,
        action_stds: torch.Tensor,
        extras: dict[str, torch.Tensor] | None = None,
        privileged_obs: dict[str, torch.Tensor] | None = None,
        adaptation_obs: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Store one step of transition data.

        All inputs are [num_envs, dim] and get stored at self.step.

        Args:
            extras: Optional dict of extra tensors to store (e.g., GT directions).
                    Keys must match names registered in __init__ extras parameter.
            privileged_obs: Optional dict of privileged obs tensors (RMA).
                    Keys must match groups registered in __init__ privileged_obs_dims.
                    Stored raw so encoders can re-run with gradients at learning time.
        """
        self.actor_obs[self.step].copy_(actor_obs)
        self.critic_obs[self.step].copy_(critic_obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards)
        self.dones[self.step].copy_(dones.unsqueeze(-1) if dones.dim() == 1 else dones)
        self.values[self.step].copy_(values if values.dim() == 2 else values.unsqueeze(-1))
        self.log_probs[self.step].copy_(
            log_probs.unsqueeze(-1) if log_probs.dim() == 1 else log_probs
        )
        self.action_means[self.step].copy_(action_means)
        self.action_stds[self.step].copy_(action_stds)

        # Store any extra tensors passed alongside standard fields
        if extras:
            for name, data in extras.items():
                self._extras[name][self.step].copy_(data)

        # Store privileged obs groups separately (RMA)
        if privileged_obs:
            for group_name, data in privileged_obs.items():
                self._privileged_obs[group_name][self.step].copy_(data)

        # Store adaptation obs (Phase 3): window flattened to [N, W*D]
        if adaptation_obs:
            for group_name, data in adaptation_obs.items():
                self._adaptation_obs[group_name][self.step].copy_(
                    data.flatten(1) if data.dim() > 2 else data
                )

        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
    ) -> None:
        """Compute GAE returns and advantages in-place (RSL-RL style).

        Args:
            last_values: Bootstrapped value for last observation [num_envs, 1]
            gamma: Discount factor
            lam: GAE lambda
            normalize_advantage: Whether to normalize advantages globally
        """
        advantage = torch.zeros(self.num_envs, 1, device=self.device)

        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + gamma * next_values * next_is_not_terminal
                - self.values[step]
            )
            advantage = delta + gamma * lam * next_is_not_terminal * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values

        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (
                self.advantages.std() + 1e-8
            )

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int,
        normalize_advantage_per_mini_batch: bool = False,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield shuffled mini-batches over all epochs.

        Flattens [T, N, dim] -> [T*N, dim] and yields random slices.
        Single random permutation reused across all epochs (RSL-RL/holosoma pattern).

        Args:
            num_mini_batches: Number of mini-batches per epoch
            num_epochs: Number of epochs to iterate
            normalize_advantage_per_mini_batch: Re-normalize advantages per batch

        Yields:
            Dict with keys: actor_obs, critic_obs, actions, returns, advantages,
            old_log_probs, old_action_means, old_action_stds, values
        """
        batch_size = self.num_envs * self.num_steps
        mini_batch_size = batch_size // num_mini_batches

        # Flatten all tensors: [T, N, dim] -> [T*N, dim]
        flat_actor_obs = self.actor_obs.flatten(0, 1)
        flat_critic_obs = self.critic_obs.flatten(0, 1)
        flat_actions = self.actions.flatten(0, 1)
        flat_returns = self.returns.flatten(0, 1)
        flat_advantages = self.advantages.flatten(0, 1)
        flat_log_probs = self.log_probs.flatten(0, 1)
        flat_action_means = self.action_means.flatten(0, 1)
        flat_action_stds = self.action_stds.flatten(0, 1)
        flat_values = self.values.flatten(0, 1)

        # Flatten extras
        flat_extras = {
            name: tensor.flatten(0, 1) for name, tensor in self._extras.items()
        }

        # Flatten privileged obs groups (RMA)
        flat_privileged_obs = {
            group: tensor.flatten(0, 1) for group, tensor in self._privileged_obs.items()
        }

        # Flatten adaptation obs groups (Phase 3)
        flat_adaptation_obs = {
            group: tensor.flatten(0, 1) for group, tensor in self._adaptation_obs.items()
        }

        # Single permutation reused across epochs (RSL-RL pattern)
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, device=self.device
        )

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                batch_idx = indices[start:end]

                mb_advantages = flat_advantages[batch_idx]
                if normalize_advantage_per_mini_batch:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                batch = {
                    "actor_obs": flat_actor_obs[batch_idx],
                    "critic_obs": flat_critic_obs[batch_idx],
                    "actions": flat_actions[batch_idx],
                    "returns": flat_returns[batch_idx],
                    "advantages": mb_advantages,
                    "old_log_probs": flat_log_probs[batch_idx],
                    "old_action_means": flat_action_means[batch_idx],
                    "old_action_stds": flat_action_stds[batch_idx],
                    "values": flat_values[batch_idx],
                }

                # Include extras in batch (shuffled with same indices)
                for name, flat_tensor in flat_extras.items():
                    batch[name] = flat_tensor[batch_idx]

                # Include privileged obs groups (RMA) — re-encoded with
                # gradients during learning, never pre-composed with actor obs
                if flat_privileged_obs:
                    batch["privileged_obs"] = {
                        group: flat_tensor[batch_idx]
                        for group, flat_tensor in flat_privileged_obs.items()
                    }

                # Include adaptation obs groups (Phase 3) — stored flattened,
                # reshaped to (B, W, D) by _compose_actor_input in RmaPPO
                if flat_adaptation_obs:
                    batch["adaptation_obs"] = {
                        group: flat_tensor[batch_idx]
                        for group, flat_tensor in flat_adaptation_obs.items()
                    }

                yield batch

    def clear(self) -> None:
        """Reset step counter for next rollout (tensors reused)."""
        self.step = 0
