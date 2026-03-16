"""Dynamic task DEFAULTS populated by @register_task decorators.

Tasks register themselves in their own __init__.py files.
This module exposes the DEFAULTS dict that gets populated at import time.
"""

# Import all tasks to trigger @register_task decorators before DEFAULTS is built
import colosseum.tasks  # noqa: F401
from colosseum.config.types.task import _TASK_REGISTRY

# DEFAULTS is a live view of the task registry
# Tasks register via @register_task in tasks/*/config/__init__.py
DEFAULTS = _TASK_REGISTRY
