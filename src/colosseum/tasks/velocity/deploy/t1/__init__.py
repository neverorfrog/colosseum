"""T1 velocity tracking deployment.

Usage (via registry):
    >>> from colosseum.deploy.core.registry import TASK_REGISTRY
    >>> from colosseum.deploy.backends.mujoco import MujocoController
    >>> config = TASK_REGISTRY.get_config("t1-velocity-rough")
    >>> with MujocoController(config) as controller:
    ...     controller.run()

Usage (direct):
    >>> from colosseum.tasks.velocity.deploy.t1 import T1_23DOF_VELOCITY_ROUGH
    >>> from colosseum.deploy.backends.mujoco import MujocoController
    >>> with MujocoController(T1_23DOF_VELOCITY_ROUGH) as controller:
    ...     controller.run()
"""

from colosseum.deploy.core.registry import register_task
from colosseum.tasks.velocity.deploy.t1.config import T1_23DOF_VELOCITY_FLAT, T1_23DOF_VELOCITY_ROUGH
from colosseum.tasks.velocity.deploy.t1.policy import T1VelocityPolicy


@register_task("t1-velocity-rough")
def t1_velocity_task():
    """T1 23-DOF velocity tracking deployment task."""
    return T1_23DOF_VELOCITY_ROUGH, T1VelocityPolicy

@register_task("t1-velocity-flat")
def t1_velocity_flat_task():
    """T1 23-DOF velocity tracking deployment task."""
    return T1_23DOF_VELOCITY_FLAT, T1VelocityPolicy

# Export for direct use
__all__ = [
    "T1_23DOF_VELOCITY_ROUGH",
    "T1_23DOF_VELOCITY_FLAT",
    "T1VelocityPolicy",
]
