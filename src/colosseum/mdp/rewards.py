"""Layer 1: Universal reward functions (robot-agnostic).

These pure functions work across all robots and tasks, providing common
reward patterns and utilities.
"""

import torch


def exponential_reward_kernel(
    error: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Exponential reward kernel for smooth error penalties.

    Returns exp(-error^2 / (2 * std^2)), which gives:
    - Reward ≈ 1.0 when error is small
    - Reward → 0 as error increases
    - Smooth gradient for learning

    Args:
        error: Error magnitude (can be scalar or vector norm).
               Shape: (N,) or scalar.
        std: Standard deviation controlling reward sharpness.
            Smaller std = sharper falloff, larger std = more tolerant.

    Returns:
        Reward in range [0, 1]. Shape matches error.

    Examples:
        >>> # Tracking error reward
        >>> vel_error = torch.tensor([0.1, 0.5, 1.0])
        >>> reward = exponential_reward_kernel(vel_error, std=0.5)
        >>> reward
        tensor([0.9608, 0.6065, 0.1353])
    """
    return torch.exp(-torch.square(error) / (2 * std**2))


def linear_tracking_reward(
    current: torch.Tensor,
    target: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Linear tracking reward using exponential kernel.

    Rewards the agent for tracking a target value (e.g., velocity command).

    Args:
        current: Current value. Shape: (N,) or (N, D).
        target: Target value. Shape: (N,) or (N, D).
        std: Standard deviation for exponential kernel.

    Returns:
        Reward in range [0, 1]. Shape: (N,).

    Examples:
        >>> # Velocity tracking
        >>> current_vel = torch.tensor([[1.0, 0.0], [0.5, 0.2]])
        >>> target_vel = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        >>> reward = linear_tracking_reward(current_vel, target_vel, std=0.5)
    """
    error = torch.norm(current - target, dim=-1)
    return exponential_reward_kernel(error, std)


__all__ = [
    "exponential_reward_kernel",
    "linear_tracking_reward",
]
