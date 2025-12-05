"""Velocity tracking tasks for humanoid robots.

This module registers velocity tracking tasks for different robot configurations.
Importing this module will register all available task variants.
"""

# Import robot-specific configs to trigger task registration
from colosseum.train.tasks.velocity.config import t1  # noqa: F401
