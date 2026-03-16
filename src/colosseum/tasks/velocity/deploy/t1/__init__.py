"""T1 velocity tracking deployment (flat terrain)."""

from colosseum.deploy.core.registry import register_task
from colosseum.tasks.velocity.deploy.t1.config import T1_23DOF_VELOCITY_FLAT
from colosseum.tasks.velocity.deploy.t1.policy import T1VelocityPolicy


@register_task("t1-velocity-flat")
def t1_velocity_flat_task():
  """T1 23-DOF velocity tracking deployment task (flat terrain)."""
  return T1_23DOF_VELOCITY_FLAT, T1VelocityPolicy


__all__ = [
  "T1_23DOF_VELOCITY_FLAT",
  "T1VelocityPolicy",
]
