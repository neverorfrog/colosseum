"""Layer 1: Universal MDP functions (robot-agnostic).

This module contains pure MDP functions that work across all robots and tasks.
These functions depend only on torch and math utilities, making them reusable
in both training and deployment contexts.
"""

from colosseum.mdp.observations import *  # noqa: F403
from colosseum.mdp.rewards import *  # noqa: F403

__all__ = [
    # Observations
    "compute_projected_gravity",
    "quat_apply_inverse",
    # Rewards
    "exponential_reward_kernel",
    "linear_tracking_reward",
]
