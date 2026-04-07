"""Task configuration types and registry for Colosseum.

Tasks register themselves via @register_task decorator in their own __init__.py files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from colosseum.config.types.algorithm import AlgorithmConfig

# Live registry: populated by @register_task decorators at import time
_TASK_REGISTRY: dict[str, "TaskConfig"] = {}


@dataclass(frozen=True)
class TaskConfig:
    """Base class for task configurations.

    Subclasses should:
    1. Set name field
    2. Override train_env_cfg property to return a ManagerBasedRlEnvCfg
    3. Override play_env_cfg property if a different config is needed for evaluation
    """

    name: str = ""

    @property
    def train_env_cfg(self):
        """Return the training environment configuration."""
        raise NotImplementedError(
            f"Task '{self.name}' must implement train_env_cfg property."
        )

    @property
    def play_env_cfg(self):
        """Return the play/evaluation environment configuration (defaults to train_env_cfg)."""
        return self.train_env_cfg

    @property
    def algo_cfg(self) -> "AlgorithmConfig | None":
        """Return the preferred algorithm configuration for this task.

        Returns None if the task has no preferred algorithm, in which case the
        global default from config/values/algorithm.py is used.
        """
        return None

    @property
    def rl_cfg(self):
        """Return the RSL-RL runner configuration (used by train_rsl_rl.py)."""
        raise NotImplementedError(
            f"Task '{self.name}' must implement rl_cfg property."
        )

    @classmethod
    def reconstruct_from_dict(cls, data: dict) -> "TaskConfig":
        """Reconstruct config from dict. Override in subclasses if needed."""
        return cls()


def register_task(name: str):
    """Decorator to register a task configuration.

    Usage:
        @register_task("my-task")
        class MyTask(TaskConfig):
            name: str = "my-task"

            @property
            def train_env_cfg(self):
                return make_my_env_cfg()

    The decorator instantiates the class and registers the instance.
    """

    def decorator(cls: type):
        instance = cls()
        _TASK_REGISTRY[name] = instance
        return cls

    return decorator


def get_task(name: str) -> TaskConfig:
    """Get a task configuration by name.

    Args:
        name: Task name (e.g., "t1-velocity-rough")

    Returns:
        TaskConfig instance

    Raises:
        KeyError: If task not found in registry
    """
    if name not in _TASK_REGISTRY:
        raise KeyError(
            f"Task '{name}' not found. "
            f"Available: {list(_TASK_REGISTRY.keys())}"
        )
    return _TASK_REGISTRY[name]


def get_task_class(name: str) -> type[TaskConfig]:
    """Get the class of a registered task by name.

    Args:
        name: Task name

    Returns:
        Task class

    Raises:
        KeyError: If task not found
    """
    task = get_task(name)
    return type(task)


def list_tasks() -> list[str]:
    """List all registered task names."""
    return list(_TASK_REGISTRY.keys())
