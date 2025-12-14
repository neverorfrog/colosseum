"""Lazy exports for deployment backends to avoid heavy optional deps."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
  "BoosterRobotController",
  "BoosterRobotPortal",
  "MujocoController",
]

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
  from .booster import BoosterRobotController, BoosterRobotPortal
  from .mujoco import MujocoController


def __getattr__(name: str):
  if name in {"BoosterRobotController", "BoosterRobotPortal"}:
    module = import_module(".booster", __name__)
  elif name == "MujocoController":
    module = import_module(".mujoco", __name__)
  else:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

  return getattr(module, name)
