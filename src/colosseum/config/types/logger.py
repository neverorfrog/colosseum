from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class LoggerConfig:
    """Configuration for experiment logging."""

    enabled: bool = True
    """Whether logging is enabled."""

    backend: str = "wandb"
    """Logging backend to use ('wandb', 'tensorboard', 'disabled')."""

    project: str = "colosseum"
    """Project name for the logging backend."""

    name: str | None = None
    """Run name. If None, will be auto-generated."""

    entity: str | None = None
    """Team/entity name for W&B. If None, uses default entity."""

    group: str | None = None
    """Group name for organizing related runs."""

    tags: tuple[str, ...] = ()
    """Tags for the run (e.g., ('baseline', 'seed42'))."""

    id: str | None = None
    """Unique run ID. If None, will be auto-generated."""

    resume: str | None = None
    """Resume mode for W&B ('allow', 'must', 'never', 'auto'). None means no resume."""

    mode: str = "online"
    """W&B mode ('online', 'offline', 'disabled')."""

    log_interval: int = 50
    """How often (in steps) to log metrics."""

    save_interval: int = 500
    """How often (in steps) to save checkpoints."""

    log_dir: str = "./logs"
    """Directory to save logs and checkpoints."""

    console_level: str = "INFO"
    """Logging level for console output (e.g., 'DEBUG', 'INFO', 'WARNING')."""

    task_name: str | None = None
    """Task name for filtering. Stored in W&B config."""

    scenario: str | None = None
    """Optional scenario name for run organization."""

    job_type: str | None = None
    """W&B job type for additional grouping dimension (e.g., algorithm name)."""
