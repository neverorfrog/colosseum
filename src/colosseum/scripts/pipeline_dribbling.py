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
    "obstacle_stage_index": 0,  # num_active=0, behavior='none'
    "warm_start_from": None,  # train from scratch
    "ball_spawn_x_range": (1.5, 3.0),  # far spawn forces the robot to walk to the ball
    "skip_phase2": True,  # no visual encoder in locomotion warmup
  },
  {
    "id": 1,
    "name": "s1_dribbling",
    "description": "dribbling with latent noise — no obstacles",
    "p1_steps": 150_000_000,
    "p2_steps": 10_000_000,
    "obstacle_stage_index": 0,
    "warm_start_from": "s0_locomotion_p1",
  },
  {
    "id": 2,
    "name": "s2_static_obstacle",
    "description": "static obstacle blocker — 1 obstacle, no motion",
    "p1_steps": 200_000_000,
    "p2_steps": 10_000_000,
    "obstacle_stage_index": 1,  # num_active=1, behavior='static_blocker'
    "warm_start_from": "s1_dribbling_p1",
  },
  {
    "id": 3,
    "name": "s3_attack_blocker",
    "description": "attack blocker — 1 obstacle, lateral blocker behavior",
    "p1_steps": 200_000_000,
    "p2_steps": 10_000_000,
    "obstacle_stage_index": 3,  # num_active=1, behavior='lateral_blocker'
    "warm_start_from": "s2_static_obstacle_p1",
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
  print(f"\n{'=' * 70}", flush=True)
  print(f"  {stage_label}", flush=True)
  print(f"{'=' * 70}", flush=True)
  print("  " + " ".join(cmd), flush=True)
  print(flush=True)
  result = subprocess.run(cmd)
  if result.returncode != 0:
    print(
      f"\n[pipeline] {stage_label} failed (exit {result.returncode}). Stopping.",
      flush=True,
    )
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
  ball_spawn_x_range: tuple[float, float] | None = None,
  extra: list[str] | None = None,
) -> list[str]:
  cmd = [
    "pixi",
    "run",
    "-e",
    "train",
    "train",
    "--logger.log-dir",
    str(log_dir),
    "--logger.name",
    run_name,
    "--learning-steps",
    str(steps),
    "--cuda",
    cuda,
    "--task.env.scene.num-envs",
    str(num_envs),
    "--task.obstacle-stage-index",
    str(obstacle_stage_index),
  ]
  if ball_spawn_x_range is not None:
    cmd += [
      "--task.ball-spawn-x-range",
      str(ball_spawn_x_range[0]),
      str(ball_spawn_x_range[1]),
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
    "pixi",
    "run",
    "-e",
    "train",
    "train-phase2",
    "--checkpoint",
    checkpoint,
    "--logger.log-dir",
    str(log_dir),
    "--logger.name",
    run_name,
    "--learning-steps",
    str(steps),
    "--cuda",
    cuda,
    "--task.env.scene.num-envs",
    str(num_envs),
    "--task.obstacle-stage-index",
    str(obstacle_stage_index),
    "--task.use-depth-camera",
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
  p.add_argument(
    "--cuda", default="0", help="GPU id(s) for training, or single id for play."
  )

  sub = p.add_subparsers(dest="command")

  # --- train subcommand (default) ---
  train_p = sub.add_parser("train", help="Run the training pipeline (default).")
  train_p.add_argument("--num-envs", type=int, default=4096)
  train_p.add_argument(
    "--p2-num-envs",
    type=int,
    default=None,
    help="Num envs for Phase 2. Defaults to --num-envs if not set.",
  )
  train_p.add_argument(
    "--start-stage",
    type=int,
    default=0,
    help="Stage to start from. Earlier checkpoints must exist.",
  )
  train_p.add_argument(
    "--end-stage",
    type=int,
    default=None,
    help="Last stage to run, inclusive. Defaults to the last stage.",
  )
  train_p.add_argument("--skip-phase2", action="store_true")
  train_p.add_argument(
    "--force",
    action="store_true",
    help="Re-run phases even if their checkpoints already exist.",
  )
  train_p.add_argument(
    "--use-dagger",
    action="store_true",
    help="Each stage imitates the previous stage's P1 checkpoint as teacher.",
  )
  train_p.add_argument(
    "--warm-start-stage",
    type=int,
    default=None,
    metavar="N",
    help=(
      "Override warm-start source for every stage being run. "
      "N is the stage id whose P1 checkpoint to load (-1 = train from scratch). "
      "Default: each stage uses its own warm_start_from definition."
    ),
  )
  train_p.add_argument(
    "--warm-start-checkpoint",
    default=None,
    metavar="PATH",
    help=(
      "Explicit checkpoint path to warm-start from. "
      "Overrides --warm-start-stage and the stage's default warm_start_from."
    ),
  )
  train_p.add_argument(
    "--teacher-stage",
    type=int,
    default=None,
    metavar="N",
    help=(
      "Stage id whose P1 checkpoint to use as DAgger teacher. "
      "Implies --use-dagger. (-1 = no teacher / plain PPO)."
    ),
  )
  train_p.add_argument(
    "--teacher-checkpoint",
    default=None,
    metavar="PATH",
    help=(
      "Explicit checkpoint path to use as DAgger teacher. "
      "Implies --use-dagger. Overrides --teacher-stage."
    ),
  )
  train_p.add_argument(
    "--pull-from",
    default=None,
    metavar="USER@HOST:PATH",
    help=(
      "Sync warm-start (and teacher) checkpoints from a remote machine before "
      "each stage. PATH is the remote log_dir "
      "(e.g. phd_student@gin:~/Maiorana/colosseum/logs/dribbling_pipeline)."
    ),
  )

  # --- play subcommand ---
  play_p = sub.add_parser(
    "play", help="Play a trained policy from the pipeline checkpoint index."
  )
  play_p.add_argument("--stage", type=int, default=0, help="Stage index to play (0–2).")
  play_p.add_argument(
    "--phase",
    type=int,
    default=1,
    choices=[1, 2],
    help="Phase to play: 1 = privileged encoder, 2 = visual adaptation encoder.",
  )
  play_p.add_argument(
    "--obstacle-stage-index",
    type=int,
    default=None,
    help="Override obstacle stage for the play env (defaults to the stage's training value).",
  )
  play_p.add_argument(
    "--sync-from",
    metavar="USER@HOST:PATH",
    default=None,
    help=(
      "Sync checkpoints from a remote training machine before playing. "
      "Example: phd_student@gin:~/Maiorana/colosseum/logs/dribbling_pipeline"
    ),
  )

  return p.parse_known_args()


def _sync_checkpoints(remote: str, log_dir: Path, stage: int, phase: int) -> None:
    """Rsync a checkpoint from a remote training machine.

    Tries two locations in order:
    1. Post-training index symlink: <remote>/checkpoints/s{stage}_p{phase}.pt
    2. Mid-run fallback: newest .pt file in <remote>/<run_name>/checkpoints/
    """
    link_name = f"s{stage}_p{phase}"
    index_dst = log_dir / "checkpoints"
    index_dst.mkdir(parents=True, exist_ok=True)
    local_ckpt = index_dst / f"{link_name}.pt"

    # --- attempt 1: post-training index symlink ---
    index_src = f"{remote}/checkpoints/{link_name}.pt"
    print(f"[pipeline] Syncing {index_src} ...", flush=True)
    result = subprocess.run(
        ["rsync", "-avz", "--copy-links", index_src, str(index_dst) + "/"],
        capture_output=True,
    )
    if result.returncode == 0:
        print(result.stdout.decode(), end="", flush=True)
        return

    # --- attempt 2: mid-run fallback via SSH find ---
    stage_def = next((s for s in STAGES if s["id"] == stage), None)
    if stage_def is None:
        print(f"[pipeline] ERROR: unknown stage {stage}", flush=True)
        sys.exit(1)

    run_name = f"{stage_def['name']}_p{phase}"

    if ":" not in remote:
        print(f"[pipeline] ERROR: --sync-from must be user@host:path, got {remote!r}", flush=True)
        sys.exit(1)
    host, remote_path = remote.split(":", 1)
    ckpt_dir = f"{remote_path}/{run_name}/checkpoints"

    print(
        f"[pipeline] Index symlink not found (training still running?); "
        f"looking for latest checkpoint in {host}:{ckpt_dir} ...",
        flush=True,
    )
    find_result = subprocess.run(
        ["ssh", host, f"ls -t {ckpt_dir}/*.pt 2>/dev/null | head -1"],
        capture_output=True, text=True,
    )
    remote_ckpt = find_result.stdout.strip()
    if find_result.returncode != 0 or not remote_ckpt:
        print(f"[pipeline] ERROR: no checkpoints found in {host}:{ckpt_dir}", flush=True)
        sys.exit(1)

    print(f"[pipeline] Syncing mid-run checkpoint {host}:{remote_ckpt} -> {local_ckpt} ...", flush=True)
    result = subprocess.run(["rsync", "-avz", "--copy-links", f"{host}:{remote_ckpt}", str(local_ckpt)])
    if result.returncode != 0:
        print(f"[pipeline] ERROR: rsync failed (exit {result.returncode}).", flush=True)
        sys.exit(result.returncode)


def _play(args: argparse.Namespace, extra_args: list[str], log_dir: Path) -> None:
  sid = args.stage
  phase = args.phase
  link_name = f"s{sid}_p{phase}"

  if getattr(args, "sync_from", None):
    _sync_checkpoints(args.sync_from, log_dir, sid, phase)

  ckpt = log_dir / "checkpoints" / f"{link_name}.pt"
  if not ckpt.exists():
    print(f"[pipeline] ERROR: no checkpoint for stage {sid} phase {phase}: {ckpt}")
    print(f"[pipeline] Run 'pipeline-dribbling train --end-stage {sid}' first.")
    sys.exit(1)

  stage_def = next((s for s in STAGES if s["id"] == sid), None)
  if stage_def is None:
    print(f"[pipeline] ERROR: unknown stage {sid}")
    sys.exit(1)

  obs_idx = (
    args.obstacle_stage_index
    if args.obstacle_stage_index is not None
    else stage_def["obstacle_stage_index"]
  )

  cmd = [
    "pixi",
    "run",
    "-e",
    "train",
    "play",
    "--checkpoint",
    str(ckpt),
    "--cuda",
    args.cuda,
    "--task.obstacle-stage-index",
    str(obs_idx),
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
      extra_args = extra_args[:idx] + extra_args[idx + 2 :]

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
    if p1_ckpt.exists() and not args.force:
      print(f"[pipeline] Stage {sid} P1 already done, skipping: {p1_ckpt}")
    else:
      warm_start_path: str | None = None
      ws_ckpt_override = getattr(args, "warm_start_checkpoint", None)
      ws_stage_override = getattr(args, "warm_start_stage", None)
      pull_from = getattr(args, "pull_from", None)

      # Pull the warm-start (and teacher) checkpoint from a remote machine.
      if pull_from and ws_ckpt_override is None and ws_stage_override != -1:
        if ws_stage_override is not None:
          pull_stage = ws_stage_override
        elif stage["warm_start_from"]:
          # Derive stage id from the built-in warm_start_from name.
          pull_stage = sid - 1
        else:
          pull_stage = None
        if pull_stage is not None:
          _sync_checkpoints(pull_from, log_dir, pull_stage, 1)

      if ws_ckpt_override is not None:
        # Explicit path — highest priority.
        ws_path = Path(ws_ckpt_override)
        if not ws_path.exists():
          print(f"[pipeline] ERROR: --warm-start-checkpoint not found: {ws_path}")
          sys.exit(1)
        warm_start_path = str(ws_path)
      elif ws_stage_override == -1:
        # Explicit "train from scratch".
        warm_start_path = None
      else:
        # Resolve from stage id override or the stage's built-in default.
        if ws_stage_override is not None:
          ws_def = next((s for s in STAGES if s["id"] == ws_stage_override), None)
          if ws_def is None:
            print(f"[pipeline] ERROR: --warm-start-stage {ws_stage_override} is not a valid stage id.")
            sys.exit(1)
          warm_start_source = ws_def["name"] + "_p1"
          fallback_link = log_dir / "checkpoints" / f"s{ws_stage_override}_p1.pt"
        else:
          warm_start_source = stage["warm_start_from"]
          fallback_link = log_dir / "checkpoints" / f"s{sid - 1}_p1.pt"

        if warm_start_source:
          ws_ckpt = _ckpt(log_dir, warm_start_source)
          if not ws_ckpt.exists():
            if fallback_link.exists():
              ws_ckpt = fallback_link.resolve()
            else:
              print(f"[pipeline] ERROR: warm-start checkpoint missing: {ws_ckpt}")
              sys.exit(1)
          warm_start_path = str(ws_ckpt)

      # Resolve teacher checkpoint for DAgger.
      tc_ckpt_override = getattr(args, "teacher_checkpoint", None)
      tc_stage_override = getattr(args, "teacher_stage", None)

      if tc_ckpt_override is not None:
        tc_path = Path(tc_ckpt_override)
        if not tc_path.exists():
          print(f"[pipeline] ERROR: --teacher-checkpoint not found: {tc_path}")
          sys.exit(1)
        teacher_path = str(tc_path)
      elif tc_stage_override == -1:
        teacher_path = None
      elif tc_stage_override is not None:
        tc_def = next((s for s in STAGES if s["id"] == tc_stage_override), None)
        if tc_def is None:
          print(f"[pipeline] ERROR: --teacher-stage {tc_stage_override} is not a valid stage id.")
          sys.exit(1)
        if pull_from:
          _sync_checkpoints(pull_from, log_dir, tc_stage_override, 1)
        tc_ckpt = _ckpt(log_dir, tc_def["name"] + "_p1")
        if not tc_ckpt.exists():
          tc_link = log_dir / "checkpoints" / f"s{tc_stage_override}_p1.pt"
          if tc_link.exists():
            tc_ckpt = tc_link.resolve()
          else:
            print(f"[pipeline] ERROR: teacher checkpoint missing: {tc_ckpt}")
            sys.exit(1)
        teacher_path = str(tc_ckpt)
      elif args.use_dagger:
        # Default DAgger behaviour: same source as warm-start.
        teacher_path = warm_start_path
      else:
        teacher_path = None

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
          ball_spawn_x_range=stage.get("ball_spawn_x_range"),
          extra=extra_args,
        ),
        f"Stage {sid} Phase 1 — {stage['description']}",
      )

    if not p1_ckpt.exists():
      print(f"[pipeline] ERROR: checkpoint missing after P1: {p1_ckpt}")
      sys.exit(1)

    _symlink_stage_ckpt(log_dir, p1_run, p1_link)

    if args.skip_phase2 or stage.get("skip_phase2", False):
      continue

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------
    p2_ckpt = _ckpt(log_dir, p2_run)
    if p2_ckpt.exists() and not args.force:
      print(f"[pipeline] Stage {sid} P2 already done, skipping: {p2_ckpt}")
    else:
      p2_envs = args.p2_num_envs if args.p2_num_envs is not None else args.num_envs
      _run(
        _pixi_train_phase2(
          log_dir=log_dir,
          run_name=p2_run,
          steps=stage["p2_steps"],
          obstacle_stage_index=stage["obstacle_stage_index"],
          cuda=args.cuda,
          num_envs=p2_envs,
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
