#!/usr/bin/env python3
"""Dribbling curriculum pipeline: Phase 1 + Phase 2 for each curriculum stage.

Stages
------
  Stage 0  —  locomotion warmup: no obstacles, ball present, 50M steps.
               Goal: stable walking before ball interaction is reinforced.
  Stage 1  —  dribbling with latent noise: no obstacles, 150M steps.
               Warm-starts from Stage 0 weights. Latent noise (latent_noise_std=0.1)
               already baked into DribblingRmaTermCfg.
  Stage 2  —  static obstacle: one static blocker, warm-starts from Stage 1 weights.
               NOTE: per-episode replay of Stage 0/1 configs is not yet implemented;
               add a MixedEnv or curriculum term to achieve it.

For each Phase 1 stage, a Phase 2 visual adaptation encoder training run follows
immediately using the Phase 1 final checkpoint.

After each stage completes, the final checkpoint is symlinked into
  <log_dir>/checkpoints/s{N}_p{1|2}.pt
for easy cross-stage reference.

Usage
-----
    pixi run -e train pipeline-dribbling
    pixi run -e train pipeline-dribbling --cuda 1
    pixi run -e train pipeline-dribbling --start-stage 1
    pixi run -e train pipeline-dribbling --log-dir ./logs/exp1 --num-envs 4096
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    {
        "id": 0,
        "name": "s0_locomotion",
        "description": "locomotion warmup — no obstacles, ball present",
        "p1_steps": 50_000_000,
        "p2_steps": 20_000_000,
        "obstacle_stage_index": 0,   # num_active=0, behavior='none'
        "warm_start_from": None,     # train from scratch
    },
    {
        "id": 1,
        "name": "s1_dribbling",
        "description": "dribbling with latent noise — no obstacles",
        "p1_steps": 150_000_000,
        "p2_steps": 30_000_000,
        "obstacle_stage_index": 0,
        "warm_start_from": "s0_locomotion_p1",
    },
    {
        "id": 2,
        "name": "s2_static_obstacle",
        "description": "static obstacle blocker — 1 obstacle, no motion",
        "p1_steps": 200_000_000,
        "p2_steps": 30_000_000,
        "obstacle_stage_index": 1,   # num_active=1, behavior='static_blocker'
        "warm_start_from": "s1_dribbling_p1",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ckpt(log_dir: Path, run_name: str) -> Path:
    return log_dir / run_name / "checkpoints" / "latest.pt"


def _symlink_stage_ckpt(log_dir: Path, run_name: str, link_name: str) -> None:
    """Create <log_dir>/checkpoints/<link_name>.pt -> ../<run_name>/checkpoints/latest.pt."""
    index_dir = log_dir / "checkpoints"
    index_dir.mkdir(parents=True, exist_ok=True)
    link_path = index_dir / f"{link_name}.pt"
    target = Path("..") / run_name / "checkpoints" / "latest.pt"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)
    print(f"[pipeline] Checkpoint index: {link_path} -> {target}", flush=True)


def _run(cmd: list[str], stage_label: str) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"  {stage_label}", flush=True)
    print(f"{'='*70}", flush=True)
    print("  " + " ".join(cmd), flush=True)
    print(flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[pipeline] {stage_label} failed (exit {result.returncode}). Stopping.", flush=True)
        sys.exit(result.returncode)


def _pixi_train(
    log_dir: Path,
    run_name: str,
    steps: int,
    obstacle_stage_index: int,
    cuda: str,
    num_envs: int,
    checkpoint: str | None = None,
    warm_start: str | None = None,
    teacher_checkpoint: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        "pixi", "run", "-e", "train", "train",
        "--logger.log-dir", str(log_dir),
        "--logger.name", run_name,
        "--learning-steps", str(steps),
        "--cuda", cuda,
        "--task.env.scene.num-envs", str(num_envs),
        "--task.obstacle-stage-index", str(obstacle_stage_index),
    ]
    if warm_start:
        cmd += ["--warm-start", warm_start]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if teacher_checkpoint:
        cmd += ["--task.use-dagger", "--task.teacher-checkpoint", teacher_checkpoint]
    cmd += extra or []
    return cmd


def _pixi_train_phase2(
    log_dir: Path,
    run_name: str,
    steps: int,
    obstacle_stage_index: int,
    cuda: str,
    num_envs: int,
    checkpoint: str,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        "pixi", "run", "-e", "train", "train-phase2",
        "--checkpoint", checkpoint,
        "--logger.log-dir", str(log_dir),
        "--logger.name", run_name,
        "--learning-steps", str(steps),
        "--cuda", cuda,
        "--task.env.scene.num-envs", str(num_envs),
        "--task.obstacle-stage-index", str(obstacle_stage_index),
    ]
    cmd += extra or []
    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Dribbling curriculum pipeline",
        epilog="Any flags after -- are forwarded verbatim to every train/play invocation.",
    )
    p.add_argument("--log-dir", default="./logs/dribbling_pipeline")
    p.add_argument("--cuda", default="0", help="GPU id(s) for training, or single id for play.")

    sub = p.add_subparsers(dest="command")

    # --- train subcommand (default) ---
    train_p = sub.add_parser("train", help="Run the training pipeline (default).")
    train_p.add_argument("--num-envs", type=int, default=4096)
    train_p.add_argument("--start-stage", type=int, default=0,
        help="Stage to start from. Earlier checkpoints must exist.")
    train_p.add_argument("--end-stage", type=int, default=None,
        help="Last stage to run, inclusive. Defaults to the last stage.")
    train_p.add_argument("--skip-phase2", action="store_true")
    train_p.add_argument("--use-dagger", action="store_true",
        help="Each stage imitates the previous stage's P1 checkpoint as teacher.")

    # --- play subcommand ---
    play_p = sub.add_parser("play", help="Play a trained policy from the pipeline checkpoint index.")
    play_p.add_argument("--stage", type=int, default=0, help="Stage index to play (0–2).")
    play_p.add_argument("--phase", type=int, default=1, choices=[1, 2],
        help="Phase to play: 1 = privileged encoder, 2 = visual adaptation encoder.")
    play_p.add_argument("--obstacle-stage-index", type=int, default=None,
        help="Override obstacle stage for the play env (defaults to the stage's training value).")

    return p.parse_known_args()


def _play(args: argparse.Namespace, extra_args: list[str], log_dir: Path) -> None:
    sid = args.stage
    phase = args.phase
    link_name = f"s{sid}_p{phase}"
    ckpt = log_dir / "checkpoints" / f"{link_name}.pt"
    if not ckpt.exists():
        print(f"[pipeline] ERROR: no checkpoint for stage {sid} phase {phase}: {ckpt}")
        print(f"[pipeline] Run 'pipeline-dribbling train --end-stage {sid}' first.")
        sys.exit(1)

    stage_def = next((s for s in STAGES if s["id"] == sid), None)
    if stage_def is None:
        print(f"[pipeline] ERROR: unknown stage {sid}")
        sys.exit(1)

    obs_idx = args.obstacle_stage_index if args.obstacle_stage_index is not None else stage_def["obstacle_stage_index"]

    cmd = [
        "pixi", "run", "-e", "train", "play",
        "--checkpoint", str(ckpt),
        "--cuda", args.cuda,
        "--task.obstacle-stage-index", str(obs_idx),
        "--task.use-depth-camera",
    ] + extra_args
    print(f"[pipeline] Playing stage {sid} phase {phase}: {ckpt}", flush=True)
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd)


def main() -> None:
    args, extra_args = parse_args()

    # --cuda may land in extra_args if placed after the subcommand name;
    # extract it so we don't emit duplicate --cuda flags in child commands.
    if "--cuda" in extra_args:
        idx = extra_args.index("--cuda")
        if idx + 1 < len(extra_args):
            args.cuda = extra_args[idx + 1]
            extra_args = extra_args[:idx] + extra_args[idx + 2:]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "play":
        _play(args, extra_args, log_dir)
        return

    end_stage = args.end_stage if args.end_stage is not None else STAGES[-1]["id"]

    for stage in STAGES:
        sid = stage["id"]
        if sid < args.start_stage or sid > end_stage:
            print(f"[pipeline] Skipping Stage {sid} ({stage['name']})")
            continue

        p1_run = f"{stage['name']}_p1"
        p2_run = f"{stage['name']}_p2"
        p1_ckpt = _ckpt(log_dir, p1_run)
        p1_link = f"s{sid}_p1"
        p2_link = f"s{sid}_p2"

        # ------------------------------------------------------------------
        # Phase 1
        # ------------------------------------------------------------------
        if p1_ckpt.exists():
            print(f"[pipeline] Stage {sid} P1 already done, skipping: {p1_ckpt}")
        else:
            warm_start_path: str | None = None
            if stage["warm_start_from"]:
                ws_ckpt = _ckpt(log_dir, stage["warm_start_from"])
                if not ws_ckpt.exists():
                    ws_link = log_dir / "checkpoints" / f"s{sid - 1}_p1.pt"
                    if ws_link.exists():
                        ws_ckpt = ws_link.resolve()
                    else:
                        print(f"[pipeline] ERROR: warm-start checkpoint missing: {ws_ckpt}")
                        sys.exit(1)
                warm_start_path = str(ws_ckpt)

            # In DAgger mode the previous stage's checkpoint is also the teacher.
            # Stage 0 always trains from scratch with plain PPO (no teacher yet).
            teacher_path = warm_start_path if (args.use_dagger and warm_start_path) else None

            _run(
                _pixi_train(
                    log_dir=log_dir,
                    run_name=p1_run,
                    steps=stage["p1_steps"],
                    obstacle_stage_index=stage["obstacle_stage_index"],
                    cuda=args.cuda,
                    num_envs=args.num_envs,
                    warm_start=warm_start_path,
                    teacher_checkpoint=teacher_path,
                    extra=extra_args,
                ),
                f"Stage {sid} Phase 1 — {stage['description']}",
            )

        if not p1_ckpt.exists():
            print(f"[pipeline] ERROR: checkpoint missing after P1: {p1_ckpt}")
            sys.exit(1)

        _symlink_stage_ckpt(log_dir, p1_run, p1_link)

        if args.skip_phase2:
            continue

        # ------------------------------------------------------------------
        # Phase 2
        # ------------------------------------------------------------------
        p2_ckpt = _ckpt(log_dir, p2_run)
        if p2_ckpt.exists():
            print(f"[pipeline] Stage {sid} P2 already done, skipping: {p2_ckpt}")
        else:
            _run(
                _pixi_train_phase2(
                    log_dir=log_dir,
                    run_name=p2_run,
                    steps=stage["p2_steps"],
                    obstacle_stage_index=stage["obstacle_stage_index"],
                    cuda=args.cuda,
                    num_envs=args.num_envs,
                    checkpoint=str(p1_ckpt),
                    extra=extra_args,
                ),
                f"Stage {sid} Phase 2 — visual adaptation encoder",
            )

        if not p2_ckpt.exists():
            print(f"[pipeline] ERROR: checkpoint missing after P2: {p2_ckpt}")
            sys.exit(1)

        _symlink_stage_ckpt(log_dir, p2_run, p2_link)

    print("\n[pipeline] All stages complete.", flush=True)
    print(f"[pipeline] Stage checkpoints index: {log_dir / 'checkpoints'}", flush=True)


if __name__ == "__main__":
    main()
