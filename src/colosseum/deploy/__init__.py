"""Deployment infrastructure for Colosseum.

This module provides core deployment infrastructure (controllers, utilities).
Task-specific deployments are located in tasks/<task>/deploy/<robot>/.

Architecture:
- deploy/core/: Base controllers and utilities (robot-agnostic)
- tasks/<task>/deploy/<robot>/: Task+robot deployment bundles

Example:
    >>> # Import task-specific deployment
    >>> from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
    >>> from colosseum.deploy.core.controllers import MujocoController
    >>>
    >>> # Run deployment
    >>> controller = MujocoController(T1_23DOF_VELOCITY_DEPLOY_CFG)
    >>> controller.run()
"""

# Import task registry utilities (kept for backwards compatibility)
from colosseum.deploy.core.utils.registry import (
    register_task,
    get_task,
    list_tasks,
)

__all__ = [
    # Task registry (optional, task bundles can be imported directly)
    "register_task",
    "get_task",
    "list_tasks",
]
