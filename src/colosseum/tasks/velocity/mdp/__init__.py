"""Velocity task MDP functions and specification.

This module contains:
- observation_spec.py: Observation contract (shared between training and deployment)
- observations.py: Training observation functions (mjlab interface)

Deployment implementations are in tasks/velocity/deploy/<robot>/
"""

from colosseum.tasks.velocity.mdp.observation_spec import (
    VELOCITY_OBS_SPEC,
    VelocityObservationSpec,
)

# Export only the spec by default to avoid hard dependency on mjlab during deploy.
__all__ = [
    "VelocityObservationSpec",
    "VELOCITY_OBS_SPEC",
]

# Optionally expose training observation helpers if mjlab is available
try:  # pragma: no cover - environment-dependent
    from colosseum.tasks.velocity.mdp.observations import *  # noqa: F403,F401
    __all__.extend([
        "base_ang_vel",
        "base_lin_vel",
        "projected_gravity",
        "joint_pos_rel",
        "joint_vel",
    ])
except Exception:  # Keep deploy lightweight if mjlab isn't installed
    pass
