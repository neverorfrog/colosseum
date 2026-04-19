"""Logging utilities for training experiments.

This module provides functions to configure loguru and W&B logging,
following the patterns from holosoma and mjlab.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import wandb
from colosseum.config.types.experiment import BaseExperimentConfig
from colosseum.config.types.logger import LoggerConfig

console = Console()


def generate_run_name(
    task_name: str,
    algo_name: str,
    scenario: str | None = None,
    seed: int | None = None,
) -> str:
    """Generate a timestamped run name for W&B.

    Args:
        task_name: Name of the task (e.g., "t1-velocity-rough")
        algo_name: Name of the algorithm (e.g., "ppo")
        scenario: Optional scenario name
        seed: Optional seed for reproducibility

    Returns:
        Run name in format: <task>_<algo>_<scenario>_<seed>_YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [task_name, algo_name]
    if scenario:
        parts.append(scenario)
    if seed is not None:
        parts.append(str(seed))
    parts.append(timestamp)

    return "_".join(parts)


def setup_loguru(
    run_dir: str | Path,
    is_main_process: bool = True,
    console_level: str | None = None,
) -> None:
    """Configure loguru for console and file logging.

    Args:
        run_dir: Directory for this specific run (will save train.log here)
        is_main_process: If False, suppresses console output (for distributed training)
        console_level: Console log level. If None, uses LOGURU_LEVEL env var or "INFO"
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add console handler (only for main process in distributed training)
    if is_main_process:
        if console_level is None:
            console_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()

        logger.add(
            sys.stdout,
            level=console_level,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>",
        )

    # Add file handler with rotation (always, even for non-main processes)
    log_file = run_dir / "train.log"
    logger.add(
        log_file,
        level="DEBUG",
        rotation="100 MB",
        retention="1 week",
        enqueue=True,  # Thread-safe
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    logger.info(f"Loguru configured. Logs saved to: {log_file}")


def save_experiment_config(
    config: BaseExperimentConfig,
    log_dir: str | Path,
    wandb_run: Any | None = None,
) -> Path:
    """Save experiment configuration to YAML and optionally upload to W&B.

    Args:
        config: Full experiment configuration to save
        log_dir: Directory where config.yaml will be saved
        wandb_run: Optional W&B run object. If provided, uploads config to W&B.

    Returns:
        Path to the saved config file
    """
    from colosseum.config.types.experiment import TrainConfig as ExperimentConfig

    if not isinstance(config, ExperimentConfig):
        logger.warning(
            f"save_experiment_config: config is {type(config)}, not TrainConfig. "
            "Skipping config save."
        )
        return Path(log_dir) / "config.yaml"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save to YAML file
    config_path = log_dir / "config.yaml"
    config.save_config(str(config_path))
    logger.info(f"Experiment config saved to: {config_path}")

    # Upload to W&B if enabled
    if wandb_run is not None:
        wandb.save(str(config_path), policy="now")

    return config_path


def setup_wandb(
    config: LoggerConfig,
    base_log_dir: str | Path,
    run_name: str,
    is_main_process: bool = True,
) -> tuple[Any | None, Path | None]:
    """Initialize W&B and return the run directory.

    Args:
        config: Logger configuration
        base_log_dir: Base directory for logs (e.g., "./logs")
        run_name: Custom run name (e.g., "t1-velocity-rough_ppo_20251230_143055")
        is_main_process: If False, skips W&B initialization (for distributed training)

    Returns:
        Tuple of (wandb.Run, run_directory_path). Both None if W&B is disabled.
    """
    if not is_main_process:
        return None, None

    # Skip if disabled
    if not config.enabled or config.mode == "disabled":
        logger.info("W&B logging disabled")
        return None, None

    base_log_dir = Path(base_log_dir)
    base_log_dir.mkdir(parents=True, exist_ok=True)

    # Configure W&B environment
    os.environ["WANDB_DIR"] = str(base_log_dir)
    os.environ["WANDB_DISABLE_CODE"] = "true"  # Don't save code snapshots
    os.environ["WANDB_SILENT"] = "true"  # Reduce console output

    if config.id:
        wandb_id = config.id
    else:
        wandb_id = run_name.replace("_", "-")  # W&B IDs can't contain underscores

    wandb_kwargs: dict[str, Any] = {
        "project": config.project,
        "name": config.name or run_name,  # Display name in W&B UI
        "id": wandb_id,  # Folder name (must be unique)
        "dir": str(base_log_dir),
        "mode": config.mode,
        "resume": "allow",  # Allow resuming if same ID is used again
    }

    # Add optional parameters
    if config.entity:
        wandb_kwargs["entity"] = config.entity
    if config.group:
        wandb_kwargs["group"] = config.group
    if config.tags:
        wandb_kwargs["tags"] = list(config.tags)
    if config.job_type:
        wandb_kwargs["job_type"] = config.job_type
    if config.resume is not None:
        wandb_kwargs["resume"] = config.resume  # Override default "allow"

    # Initialize W&B
    run = wandb.init(**wandb_kwargs)

    if run is not None:
        run_dir = Path(run.dir).parent  # run.dir points to files/, we want parent
        logger.info(f"W&B initialized: {run.url}")
        logger.info(f"Run directory: {run_dir}")
        return run, run_dir
    else:
        logger.warning("W&B initialization returned None")
        return None, None


def log_metrics(metrics: dict[str, float | int | torch.Tensor], step: int) -> None:
    """Log metrics to W&B.

    Args:
        metrics: Dictionary of metric names to values
        step: Global training step
    """
    if wandb.run is None:
        return

    # Convert tensors to Python scalars
    processed_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            processed_metrics[key] = value.item()
        else:
            processed_metrics[key] = value

    # Log to W&B
    wandb.log(processed_metrics, step=step)


def extract_episode_metrics(
    log_dict: dict[str, torch.Tensor | float | int]
) -> dict[str, float]:
    """Extract episode metrics from mjlab's info["log"] dict.

    mjlab's RewardManager returns metrics like:
        {"Episode_Reward/reach_goal": tensor(0.856), ...}

    This function converts them to Python scalars for logging.

    Args:
        log_dict: The info["log"] dict from env.step() or env.reset()

    Returns:
        Dictionary with same keys but Python float values
    """
    extracted = {}
    for key, value in log_dict.items():
        if isinstance(value, torch.Tensor):
            extracted[key] = value.item()
        else:
            extracted[key] = float(value)
    return extracted


def teardown_wandb() -> None:
    """Cleanly finish W&B run."""
    try:
        if wandb.run is not None:
            logger.info("Finishing W&B run...")
            wandb.finish()
    except ImportError:
        pass


@contextmanager
def record_time():
    """Context manager to track execution time."""
    start = time.perf_counter()

    def get_elapsed():
        return time.perf_counter() - start

    yield get_elapsed


def log_training_step(
    step: int,
    total_steps: int,
    loss_dict: dict[str, float],
    episode_metrics: dict[str, float] | None = None,
    phase: int | None = None,
    collection_time: float = 0.0,
    learning_time: float = 0.0,
    elapsed_time: float = 0.0,
    num_envs: int = 1,
    log_interval: int = 1,
    title: str = "Training Progress",
    use_rich: bool = True,
    steps_per_log_step: int = 1,
) -> None:
    """Log training step to console only (Rich or loguru).

    This function provides CONSOLE OUTPUT ONLY. W&B logging should be handled
    separately via log_fn passed to the algorithm. This avoids double logging
    and prefix confusion.

    Args:
        step: Current training step
        total_steps: Total number of training steps
        loss_dict: Dictionary of training losses (averaged over log_interval)
        episode_metrics: Optional episode metrics from environment (already prefixed)
        phase: Optional training phase identifier (e.g. 1 or 2)
        collection_time: Time spent collecting data since last log (seconds)
        learning_time: Time spent learning since last log (seconds)
        elapsed_time: Total time elapsed since training started (seconds)
        num_envs: Number of parallel environments
        log_interval: Number of steps between logs (for FPS calculation)
        title: Title for the output panel
        use_rich: If True, use Rich panels; if False, use loguru (default: True)
    """
    # Calculate FPS (total samples / total time)
    total_time = collection_time + learning_time
    total_samples = num_envs * log_interval * steps_per_log_step
    fps = total_samples / total_time if total_time > 0 else 0

    obstacle_stage_name = None
    if episode_metrics is not None:
        obstacle_stage_idx = episode_metrics.get(
            "Curriculum/obstacle/obstacle_stage_index"
        )
        if obstacle_stage_idx is None:
            obstacle_stage_idx = episode_metrics.get("Curriculum/obstacle_stage_index")
        if obstacle_stage_idx is not None:
            obstacle_stage_name = {
                0: "none",
                1: "static_blocker",
                2: "lateral_blocker",
                3: "ball_attacker",
                4: "mixed_attackers",
            }.get(int(round(obstacle_stage_idx)), "unknown")

    # Console output - choose between Rich and loguru
    if use_rich:
        # Rich panel output (clean, visual)
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Progress header
        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0
        header = Text()
        header.append(f"Step {step:,}/{total_steps:,}", style="bold white")
        header.append(f" ({progress_pct:.1f}%)", style="dim white")

        # Performance metrics
        table.add_row("", "")  # Spacer
        table.add_row("Performance", "", style="bold yellow")
        table.add_row("  FPS", f"{fps:,.0f} steps/s")
        table.add_row("  Interval Time", f"{total_time:.3f}s")
        table.add_row("    - Collection", f"{collection_time:.3f}s")
        table.add_row("    - Learning", f"{learning_time:.3f}s")

        # Format elapsed time nicely (hours:minutes:seconds)
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            elapsed_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
        elif minutes > 0:
            elapsed_str = f"{minutes:02d}m {seconds:02d}s"
        else:
            elapsed_str = f"{seconds:.1f}s"
        table.add_row("  Elapsed", elapsed_str, style="bold green")

        # Training losses
        if loss_dict:
            table.add_row("", "")  # Spacer
            table.add_row("Training Losses", "", style="bold yellow")
            for k, v in loss_dict.items():
                table.add_row(f"  {k}", f"{v:.6f}")

        # Episode metrics
        if episode_metrics:
            table.add_row("", "")  # Spacer
            table.add_row("Episode Metrics", "", style="bold yellow")
            if phase is not None:
                table.add_row("  Training/phase", str(phase))
            if obstacle_stage_name is not None:
                table.add_row("  Curriculum/obstacle_stage", obstacle_stage_name)
            for k, v in episode_metrics.items():
                # Format based on magnitude
                if abs(v) < 0.01:
                    formatted = f"{v:.6f}"
                elif abs(v) < 10:
                    formatted = f"{v:.4f}"
                else:
                    formatted = f"{v:.2f}"
                table.add_row(f"  {k}", formatted)

        # Create panel
        panel = Panel(
            table,
            title=f"[bold]{title}[/bold]",
            subtitle=header,
            border_style="blue",
            padding=(1, 2),
        )

        # Print to console
        console.print(panel)
    else:
        # Loguru text output (traditional)
        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0

        # Build log message
        lines = []
        lines.append("=" * 80)
        lines.append(f"Step {step:,}/{total_steps:,} ({progress_pct:.1f}%)")
        lines.append("-" * 80)
        lines.append(
            f"Performance: {fps:,.0f} steps/s | Collection: {collection_time:.3f}s | Learning: {learning_time:.3f}s"
        )

        if loss_dict:
            lines.append("-" * 80)
            lines.append("Training Losses:")
            for k, v in loss_dict.items():
                lines.append(f"  {k:.<30} {v:.6f}")

        if episode_metrics:
            lines.append("-" * 80)
            lines.append("Episode Metrics:")
            if phase is not None:
                lines.append(f"  {'Training/phase':.<30} {phase}")
            if obstacle_stage_name is not None:
                lines.append(f"  {'Curriculum/obstacle_stage':.<30} {obstacle_stage_name}")
            for k, v in episode_metrics.items():
                if abs(v) < 0.01:
                    formatted = f"{v:.6f}"
                elif abs(v) < 10:
                    formatted = f"{v:.4f}"
                else:
                    formatted = f"{v:.2f}"
                lines.append(f"  {k:.<30} {formatted}")

        lines.append("=" * 80)

        # Log as single multiline message
        logger.info("\n" + "\n".join(lines))
