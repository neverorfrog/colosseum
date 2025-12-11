"""Layer 2: Booster T1 robot-specific MDP functions.

This module contains MDP functions specific to the Booster T1 humanoid robot.
These functions work in both training and deployment contexts and build on
Layer 1 universal functions.
"""

from colosseum.robots.booster_t1.mdp.observations import *  # noqa: F403

__all__ = [
    # Observations
    "compute_foot_contact_state",
]
