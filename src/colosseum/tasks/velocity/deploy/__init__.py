"""Velocity task deployment configurations.

This module contains deployment implementations for velocity tracking on different robots.
Each subdirectory contains a complete deployment bundle (robot config + policy + config).

Available deployments:
- t1_23dof: T1 full body (23 DOF)
"""

# Import deployment bundles
from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG

__all__ = [
    "T1_23DOF_VELOCITY_DEPLOY_CFG",
]
