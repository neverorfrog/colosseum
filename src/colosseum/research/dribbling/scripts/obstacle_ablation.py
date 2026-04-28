#!/usr/bin/env python3
"""Obstacle ablation: fill Table 5 (success-rate across curriculum stages).

Evaluates s1_p1 / s2_p1 / s3_p1 checkpoints under three obstacle conditions
(no obstacles / static blocker / ball attacker) and prints the 3×3 table.

50 episodes per cell — 10 parallel envs × 5 seeds.

Usage:
    pixi run -e train python -m colosseum.scripts.ablation_obstacle
    pixi run -e train python -m colosseum.scripts.ablation_obstacle --cuda 1
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path

import torch
from colosseum.utils.train.env import make_env
from loguru import logger
from mjlab.utils.torch import configure_torch_backends

import colosseum.tasks  # noqa: F401 — populate task registry
from colosseum.config.types.task import get_task
from colosseum.tasks.dribbling.config.t1_23dof.t1_dribbling_cfg import (
  booster_t1_dribbling_env_cfg,
)
from colosseum.utils.torch import get_device

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]  # …/colosseum/
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
CHECKPOINTS = {
  "Stage 1": CHECKPOINTS_DIR / "s1_p1.pt",
  "Stage 2": CHECKPOINTS_DIR / "s2_p1.pt",
  "Stage 3": CHECKPOINTS_DIR / "s3_p1.pt",
}

# obstacle_stage_index values from cact_cfg.py curriculum stages:
#   0 → num_active=0, behavior="none"          (no obstacles)
#   1 → num_active=1, behavior="static_blocker"
#   3 → num_active=1, behavior="ball_attacker"
EVAL_CONDITIONS: dict[str, int] = {
  "No obstacles": 0,
  "Static obstacle": 1,
  "Ball attacker": 3,
}

TARGET_REACHED_THRESHOLD = 0.75  # metres
EPISODE_TIME_BUDGET_S = 20.0  # seconds per episode
EPISODES_PER_SEED = 10
NUM_SEEDS = 5
NUM_ENVS = EPISODES_PER_SEED  # one episode per env per seed

# Policy runs at 50 Hz (policy_dt=0.02 s). Steps per episode = budget / dt.
POLICY_DT = 0.02
MAX_EPISODE_STEPS = int(EPISODE_TIME_BUDGET_S / POLICY_DT) + 50  # +buffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_env_cfg(obstacle_stage_index: int):
  cfg = booster_t1_dribbling_env_cfg(
    play=True,
    obstacle_stage_index=obstacle_stage_index,
  )
  # Restore a finite episode length (play=True sets it to ∞)
  cfg = replace(cfg, episode_length_s=EPISODE_TIME_BUDGET_S)
  # Set number of parallel envs
  cfg = replace(cfg, scene=replace(cfg.scene, num_envs=NUM_ENVS))
  return cfg


def _load_agent(checkpoint_path: Path, env, device):
  task = get_task("t1-dribbling")
  algo_cfg = task.algo_cfg

  module_path, class_name = algo_cfg.target.rsplit(":", 1)
  algo_class = getattr(importlib.import_module(module_path), class_name)

  algo = algo_class(
    config=algo_cfg,
    env=env,
    device=device,
    log_fn=lambda _m, _s: None,
    log_interval=-1,
  )
  state = algo.load(checkpoint_path)
  logger.info(f"  Checkpoint loaded — step {state.get('global_step', 0):,}")

  algo.actor.eval()
  algo.actor_obs_normalizer.eval()
  if hasattr(algo, "rma_manager"):
    algo.rma_manager.eval()

  def agent_fn(obs_dict):
    with torch.no_grad():
      actor_obs = algo.get_actor_obs(obs_dict)
      norm_obs = algo.actor_obs_normalizer(actor_obs)
      priv_obs = (
        algo.get_privileged_obs(obs_dict) if hasattr(algo, "get_privileged_obs") else {}
      )
      composed = algo._compose_actor_input(norm_obs, priv_obs)
      return algo._eval_get_action(composed)

  return agent_fn


def _run_seed(env, agent_fn, device: torch.device) -> int:
  """Run one round of NUM_ENVS episodes. Returns number of successes.

  Success = ball came within TARGET_REACHED_THRESHOLD metres of the target
  at least once during the episode. We check distance directly rather than
  relying on the command's target_reached_mask (which gets cleared before
  env.step() returns) or on the termination threshold config.
  """
  obs, _ = env.reset()
  finished = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
  ever_reached = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
  successes = 0
  steps = 0

  ball_vel_term = env.command_manager.get_term("ball_vel")

  while not finished.all() and steps < MAX_EPISODE_STEPS:
    actions = agent_fn(obs)
    obs, _, terminated, truncated, _ = env.step(actions)
    steps += 1

    # Direct distance check — threshold is always exactly TARGET_REACHED_THRESHOLD,
    # independent of what the command/termination configs use internally.
    ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]  # (N, 2)
    target_pos = ball_vel_term.target_position  # (N, 2)
    dist = (target_pos - ball_pos).norm(dim=-1)  # (N,)
    ever_reached |= (dist <= TARGET_REACHED_THRESHOLD) & ~finished

    newly_done = (terminated | truncated) & ~finished
    if newly_done.any():
      successes += int((newly_done & ever_reached).sum().item())
      finished |= newly_done
      ever_reached &= ~newly_done  # clear for envs starting fresh episodes

  return successes


def _evaluate_cell(ckpt_path: Path, obstacle_stage_index: int, device) -> float:
  env_cfg = _build_env_cfg(obstacle_stage_index)
  env = make_env(env_cfg, str(device), render_mode=None)
  try:
    agent_fn = _load_agent(ckpt_path, env, device)
    total_successes = 0
    for seed in range(NUM_SEEDS):
      count = _run_seed(env, agent_fn, device)
      logger.debug(f"    seed {seed + 1}/{NUM_SEEDS}: {count}/{EPISODES_PER_SEED}")
      total_successes += count
  finally:
    env.close()

  total_episodes = NUM_SEEDS * EPISODES_PER_SEED
  return total_successes / total_episodes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(cuda: bool = True, device_id: int = 0) -> None:
  configure_torch_backends()
  device = get_device(cuda=cuda, device_id=device_id)

  cond_names = list(EVAL_CONDITIONS.keys())
  ckpt_names = list(CHECKPOINTS.keys())

  # results[ckpt_name][cond_name] = success_rate
  results: dict[str, dict[str, float]] = {k: {} for k in ckpt_names}

  for ckpt_name, ckpt_path in CHECKPOINTS.items():
    if not ckpt_path.exists():
      logger.warning(f"Checkpoint not found, skipping: {ckpt_path}")
      for cond in cond_names:
        results[ckpt_name][cond] = float("nan")
      continue

    for cond_name, stage_idx in EVAL_CONDITIONS.items():
      logger.info(f"\n[{ckpt_name}] {cond_name} (obstacle_stage={stage_idx})")
      rate = _evaluate_cell(ckpt_path, stage_idx, device)
      results[ckpt_name][cond_name] = rate
      logger.success(f"  → {rate:.1%}")

  # ── pretty-print the table ──────────────────────────────────────────────
  col_w = 10
  sep = "-" * (20 + col_w * len(ckpt_names))
  header = f"\n{'Condition':<20}" + "".join(f"{n:>{col_w}}" for n in ckpt_names)
  print(header)
  print(sep)
  for cond in cond_names:
    row = f"{cond:<20}"
    for ckpt in ckpt_names:
      v = results[ckpt][cond]
      row += f"{'n/a':>{col_w}}" if v != v else f"{v:>{col_w - 1}.1%} "
    print(row)
  print(sep)
  print(
    f"\n(threshold={TARGET_REACHED_THRESHOLD} m, {NUM_SEEDS}×{EPISODES_PER_SEED} eps/cell)\n"
  )


if __name__ == "__main__":
  import argparse

  logger.remove()
  logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="DEBUG",
  )

  p = argparse.ArgumentParser()
  p.add_argument("--cuda", type=int, default=0, help="GPU id (-1 for CPU)")
  args = p.parse_args()

  main(cuda=args.cuda >= 0, device_id=max(args.cuda, 0))
