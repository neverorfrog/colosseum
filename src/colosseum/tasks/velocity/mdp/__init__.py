"""Velocity task MDP functions and specification.

This module contains:
- observation_spec.py: Observation contract (shared between training and deployment)
- observations.py: Training observation functions (mjlab interface)

Deployment implementations are in tasks/velocity/deploy/<robot>/
"""

from colosseum.tasks.velocity.mdp.observation_spec import (
    VelocityObservationSpec,
    VELOCITY_OBS_SPEC,
)
from colosseum.tasks.velocity.mdp.observations import *  # noqa: F403

__all__ = [
    # Specification (shared contract)
    "VelocityObservationSpec",
    "VELOCITY_OBS_SPEC",
    # Training observation functions
    "base_ang_vel",
    "base_lin_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel",
]
