"""T1 23-DOF velocity tracking deployment.

This module registers the T1 velocity tracking task and provides the configuration
for direct use if needed.

Usage (via registry):
    >>> from colosseum.deploy.core.registry import TASK_REGISTRY
    >>> from colosseum.deploy.backends.mujoco import MujocoController
    >>> config = TASK_REGISTRY.get_config("t1-velocity")
    >>> with MujocoController(config) as controller:
    ...     controller.run()

Usage (direct):
    >>> from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY
    >>> from colosseum.deploy.backends.mujoco import MujocoController
    >>> with MujocoController(T1_23DOF_VELOCITY) as controller:
    ...     controller.run()

Override example:
    >>> from dataclasses import replace
    >>> custom_cfg = replace(
    ...     T1_23DOF_VELOCITY,
    ...     policy=replace(
    ...         T1_23DOF_VELOCITY.policy,
    ...         checkpoint_path="deploy/models/custom_policy.onnx",
    ...         export_checkpoint_path="wandb/run-XYZ/files/model.pt",
    ...     ),
    ... )
"""

from colosseum.deploy.core.registry import register_task
from colosseum.tasks.velocity.deploy.t1_23dof.config import T1_23DOF_VELOCITY
from colosseum.tasks.velocity.deploy.t1_23dof.policy import T1VelocityPolicy


# Register the T1 velocity tracking task
# Simple! Just return the config and policy class - no factory needed
@register_task("t1-velocity")
def t1_velocity_task():
    """T1 23-DOF velocity tracking deployment task."""
    return T1_23DOF_VELOCITY, T1VelocityPolicy


# Export for direct use
__all__ = [
    "T1_23DOF_VELOCITY",
    "T1VelocityPolicy",
]
