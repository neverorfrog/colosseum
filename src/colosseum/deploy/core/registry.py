"""Task registry for deployment system.

This module provides a registry for deployment tasks, storing complete
ControllerConfig objects and their associated Policy classes.
"""

from __future__ import annotations
import importlib
from pathlib import Path
from typing import Dict, Tuple, Callable
from colosseum.deploy.config import ControllerConfig
from colosseum.deploy.core.policy import Policy


class TaskRegistry:
    """Registry for deployment tasks.

    Maps task names (e.g., "t1-velocity") to complete ControllerConfig objects,
    and policy types to Policy classes. This enables:
    1. Users to select tasks by name (CLI, programmatic)
    2. BaseController to instantiate policies using registered classes

    Example:
        @register_task("t1-velocity")
        def t1_velocity_task():
            return T1_23DOF_VELOCITY, T1VelocityPolicy

        # Later, in user code:
        config = TASK_REGISTRY.get_config("t1-velocity")
        controller = MujocoController(config)  # Creates instance from blueprint

        # Inside BaseController.__init__:
        policy_class = TASK_REGISTRY.get_policy(cfg.policy.task_name)
        self.policy = policy_class(self)  # Direct instantiation
    """

    def __init__(self):
        self._tasks: Dict[str, ControllerConfig] = {}
        self._policies: Dict[str, type[Policy]] = {}

    def register(
        self,
        task_name: str,
        config: ControllerConfig,
        policy_class: type[Policy]
    ) -> None:
        """Register a deployment task.

        Args:
            task_name: Task identifier (e.g., "t1-velocity")
            config: Complete controller configuration (static blueprint)
            policy_class: Policy class with __init__(controller: BaseController)

        Raises:
            ValueError: If task_name is already registered
        """
        if task_name in self._tasks:
            raise ValueError(
                f"Task '{task_name}' is already registered. "
                f"Each task can only be registered once."
            )

        self._tasks[task_name] = config

        # Index policy class by policy_type for BaseController lookup
        policy_type = config.policy.task_name
        if policy_type in self._policies:
            # Allow same policy type for multiple tasks
            # (e.g., t1-velocity, t1-velocity-rough-terrain use same policy class)
            pass
        self._policies[policy_type] = policy_class

    def get_config(self, task_name: str) -> ControllerConfig:
        """Get configuration blueprint by task name.

        Used by CLI and user code to retrieve a preset configuration.
        Users then create BaseController instances from this blueprint.

        Args:
            task_name: Task identifier (e.g., "t1-velocity")

        Returns:
            Complete controller configuration (frozen dataclass)

        Raises:
            ValueError: If task_name is not registered
        """
        if task_name not in self._tasks:
            available = list(self._tasks.keys())
            raise ValueError(
                f"Unknown task: '{task_name}'. "
                f"Available tasks: {available}. "
                f"Did you forget to register the task with @register_task()?"
            )
        return self._tasks[task_name]

    def get_policy(self, task_name: str) -> type[Policy]:
        """Get policy class by policy type.

        Used by BaseController to instantiate policies.

        Args:
            task_name: Policy type from PolicyConfig.task_name

        Returns:
            Policy class (subclass of Policy ABC)

        Raises:
            ValueError: If policy_type is not registered
        """
        if task_name not in self._policies:
            available = list(self._policies.keys())
            raise ValueError(
                f"Unknown policy type: '{task_name}'. "
                f"Available types: {available}. "
                f"Did you forget to register a task using this policy type?"
            )
        return self._policies[task_name]
    
    def list_tasks(self) -> Dict[str, ControllerConfig]:
        """List all registered tasks.

        Returns:
            Dictionary mapping task names to configurations
        """
        return dict(self._tasks)

    def list_policy_types(self) -> list[str]:
        """List all registered policy types.

        Returns:
            List of registered policy type strings
        """
        return list(self._policies.keys())


# Global registry
TASK_REGISTRY = TaskRegistry()


def register_task(task_name: str):
    """Decorator to register a deployment task.

    The decorated function should return a tuple of:
    (ControllerConfig, type[Policy])

    Args:
        task_name: Task identifier (e.g., "t1-velocity")

    Returns:
        Decorator function

    Example:
        @register_task("t1-velocity")
        def t1_velocity_task():
            # No factory needed - just return config and policy class!
            return T1_23DOF_VELOCITY, T1VelocityPolicy
    """
    def decorator(func: Callable[[], Tuple[ControllerConfig, type[Policy]]]):
        config, policy_class = func()
        TASK_REGISTRY.register(task_name, config, policy_class)
        return func
    return decorator


def auto_register_tasks() -> None:
    """Automatically import all task deployment modules.

    Scans the tasks directory for all deploy/*/__init__.py files and imports them,
    triggering their @register_task decorators.

    Pattern: tasks/{task_name}/deploy/{robot_config}/__init__.py
    Example: tasks/velocity/deploy/t1/__init__.py
             → colosseum.tasks.velocity.deploy.t1
    """
    # Get tasks directory (relative to this file: deploy/core/registry.py)
    # Go up to src/colosseum, then down to tasks
    deploy_core_dir = Path(__file__).parent  # deploy/core
    colosseum_dir = deploy_core_dir.parent.parent  # src/colosseum
    tasks_dir = colosseum_dir / "tasks"

    if not tasks_dir.exists():
        print(f"[AutoRegister] Warning: tasks directory not found: {tasks_dir}")
        return

    # Find all tasks/*/deploy/*/__init__.py files
    task_modules = list(tasks_dir.glob("*/deploy/*/__init__.py"))

    if not task_modules:
        print(f"[AutoRegister] Warning: No task deployment modules found in {tasks_dir}")
        return

    registered_count = 0
    for task_module_path in task_modules:
        # Extract module path from file path
        # tasks/velocity/deploy/t1/__init__.py
        # → ['colosseum', 'tasks', 'velocity', 'deploy', 't1_23dof']
        relative_path = task_module_path.relative_to(colosseum_dir.parent)
        parts = list(relative_path.parts[:-1])  # Remove __init__.py

        # Build module path: colosseum.tasks.velocity.deploy.t1
        module_path = ".".join(parts)

        try:
            importlib.import_module(module_path)
            registered_count += 1
        except Exception as e:
            print(f"[AutoRegister] Warning: Failed to import {module_path}: {e}")

    print(f"[AutoRegister] Imported {registered_count} task deployment modules")

