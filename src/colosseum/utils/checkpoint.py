"""Checkpoint resolution utilities shared across scripts."""

from __future__ import annotations

from pathlib import Path

from colosseum.algorithm.base_algorithm import get_latest_checkpoint


def resolve_checkpoint(
    checkpoint: str | None = None,
    run_name: str | None = None,
    log_dir: str | Path = "./logs",
) -> Path | None:
    """Resolve a checkpoint path.

    Priority:
      1. ``checkpoint`` — explicit file or directory (uses latest .pt if a dir).
      2. ``run_name``   — looks in ``<log_dir>/<run_name>/checkpoints/``.
      3. Auto-discover  — scans ``<log_dir>/`` for the most-recently modified
                          run directory that contains checkpoints.

    Returns ``None`` when nothing is found.
    """
    log_dir = Path(log_dir)

    if checkpoint and str(checkpoint).lower() != "latest":
        p = Path(checkpoint)
        if p.is_dir():
            return get_latest_checkpoint(p)
        return p if p.exists() else None

    if run_name:
        ckpt_dir = log_dir / run_name / "checkpoints"
        ckpt = get_latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            return ckpt

    if log_dir.exists():
        run_dirs = sorted(
            [d for d in log_dir.iterdir() if d.is_dir() and (d / "checkpoints").exists()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs:
            ckpt = get_latest_checkpoint(run_dir / "checkpoints")
            if ckpt is not None:
                return ckpt

    return None
