"""HumanoidSoccerMaze benchmark evaluation runner.

Implements Stage 1 (goal generalization) and Stage 2 (map generalization)
evaluation as described in the paper.  Loads training checkpoints, evaluates
deterministic policies on held-out goals and environment seeds, and computes
Task Success Rate (TSR) and Sample Efficiency Score (SES).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from mjlab.envs import ManagerBasedRlEnv
from tqdm import tqdm

import colosseum.tasks  # noqa: F401  — populate task registry
from colosseum.mdp.abstraction.maze.goal_command import MazeGoalCommandCfg
from colosseum.tasks.soccer_maze.config.t1_23dof import t1_soccer_maze_env_cfg

from .benchmark_config import BenchmarkConfig

# ── helpers ─────────────────────────────────────────────────────────────────────


def _find_checkpoints(log_dir: Path) -> list[tuple[int, Path]]:
  """Return sorted (global_step, path) tuples for all checkpoint files in a dir."""
  ckpt_dir = log_dir / "checkpoints"
  if not ckpt_dir.exists():
    return []
  entries: list[tuple[int, Path]] = []
  for p in ckpt_dir.glob("model_*.pt"):
    try:
      step = int(p.stem.split("_")[1])
      entries.append((step, p))
    except (IndexError, ValueError):
      pass
  entries.sort(key=lambda x: x[0])
  return entries


def _make_eval_env(
  map_name: str,
  goal_local: torch.Tensor,
  device: str,
  episode_length_s: float = 120.0,
) -> ManagerBasedRlEnv:
  """Create a single-env ColosseumEnv with a specific goal.

  Args:
      map_name: Maze layout key (e.g. "umaze").
      goal_local: [1, 2] or [2] tensor with goal position in LOCAL coordinates.
      device: Torch device string.
      episode_length_s: Episode timeout.
  """
  if goal_local.dim() == 1:
    goal_local = goal_local.unsqueeze(0)

  cfg = t1_soccer_maze_env_cfg(scenario=map_name, num_envs=1, play=False)
  cfg = replace(
    cfg,
    episode_length_s=episode_length_s,
    scene=replace(cfg.scene, num_envs=1),
    commands={
      **cfg.commands,
      "goal": MazeGoalCommandCfg(
        static_goals=True,
        goals=goal_local,
      ),
    },
  )
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}

  env = cfg.class_type(cfg=cfg, device=device)
  return env


def _load_ppo_policy(
  checkpoint_path: Path,
  env: ManagerBasedRlEnv,
  device: torch.device,
) -> PPO:
  """Load a PPO checkpoint and return the algorithm with actor in eval mode."""
  from importlib import import_module

  algo_cfg = _get_ppo_config()
  module_path, class_name = algo_cfg.target.rsplit(":", 1)
  module = import_module(module_path)
  algo_class = getattr(module, class_name)

  algo = algo_class(
    config=algo_cfg,
    env=env,
    device=device,
    log_fn=lambda _m, _s: None,
    log_interval=-1,
  )
  algo.load(checkpoint_path)
  algo.actor.eval()
  algo.actor_obs_normalizer.eval()
  return algo


def _get_ppo_config() -> PpoConfig:
  from colosseum.tasks.soccer_maze.config.t1_23dof.algo_cfg import (
    t1_soccer_maze_ppo_cfg,
  )
  return t1_soccer_maze_ppo_cfg()


def _run_episode(
  algo: PPO,
  env: ManagerBasedRlEnv,
) -> bool:
  """Run one deterministic episode.  Returns True if ball reached goal."""
  obs_dict, _ = env.reset()

  term_mgr = getattr(env.unwrapped, "termination_manager", None)
  success_term = "arrived_at_goal"

  while True:
    with torch.no_grad():
      actor_obs = algo.get_actor_obs(obs_dict)
      normalized_obs = algo.actor_obs_normalizer(actor_obs)
      actions = algo.actor.act_inference(normalized_obs)

    obs_dict, _rewards, terminated, truncated, _infos = env.step(actions)

    if terminated.is_floating_point():
      dones = (terminated + truncated.float()).clamp(0.0, 1.0) >= 0.5
    else:
      dones = terminated | truncated

    if dones.any():
      if term_mgr is not None and success_term in term_mgr.active_terms:
        success_flags = term_mgr.get_term(success_term)
      else:
        success_flags = terminated.bool() & ~truncated
      return bool(success_flags[0].item())


# ── main evaluation ─────────────────────────────────────────────────────────────


def evaluate_stage1(
  bench_cfg: BenchmarkConfig,
  logs_root: Path,
  output_dir: Path,
  device: str = "cuda:0",
) -> dict[str, Any]:
  """Run Stage 1 evaluation: goal generalization on a fixed map.

  Args:
      bench_cfg: Fully-built BenchmarkConfig.
      logs_root: Root directory containing per-seed training log dirs.
      output_dir: Where to write results JSON.
      device: Torch device for evaluation.

  Returns:
      Dict with TSR, SES, and per-run details.
  """
  map_name = bench_cfg.test_map
  test_goals = bench_cfg.test_goals(map_name)  # [num_test_goals, 2]

  device_obj = torch.device(device)

  # Per-agent-seed directories (matching train script naming)
  run_dirs = [
    logs_root / f"benchmark_{map_name}_agentseed{s}"
    for s in bench_cfg.agent_seeds
  ]

  # ── Collect all checkpoints per agent seed ─────────────────────────────────
  all_checkpoints: list[tuple[int, int, Path]] = []
  for agent_idx, run_dir in enumerate(run_dirs):
    for step, path in _find_checkpoints(run_dir):
      all_checkpoints.append((agent_idx, step, path))
  all_checkpoints.sort(key=lambda x: x[1])  # sort by step

  if not all_checkpoints:
    logger.error(f"No checkpoints found under {logs_root}")
    return {"error": "no checkpoints"}

  logger.info(f"Found {len(all_checkpoints)} checkpoint(s) across {len(run_dirs)} runs")

  # ── Evaluation state ───────────────────────────────────────────────────────
  # success_map[agent_idx][goal_idx][seed_idx] = (earliest_step, success_bool)
  num_agents = bench_cfg.num_agent_seeds
  num_goals = bench_cfg.num_test_goals
  num_seeds = bench_cfg.num_test_env_seeds

  earliest_success: list[list[list[int | None]]] = [
    [[None for _ in range(num_seeds)] for _ in range(num_goals)]
    for _ in range(num_agents)
  ]

  # Load checkpoint once per (agent_idx, step) pair and evaluate all
  # remaining (goal, seed) pairs.
  evaled_steps: set[tuple[int, int]] = set()

  for agent_idx, step, ckpt_path in all_checkpoints:
    key = (agent_idx, step)
    if key in evaled_steps:
      continue
    evaled_steps.add(key)

    # Check if any unevaluated (goal, seed) remain for this agent
    remaining = 0
    for g in range(num_goals):
      for s in range(num_seeds):
        if earliest_success[agent_idx][g][s] is None:
          remaining += 1
    if remaining == 0:
      continue

    # Create a "dummy" env to load the policy (env dims must match checkpoint)
    dummy_env = _make_eval_env(
      map_name,
      test_goals[0],
      device=device,
      episode_length_s=bench_cfg.episode_length_s,
    )

    try:
      algo = _load_ppo_policy(ckpt_path, dummy_env, device_obj)
    except Exception as e:
      logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
      continue

    logger.info(
      f"[agent={agent_idx} step={step:012d}] "
      f"evaluating {remaining} remaining (goal, seed) pairs"
    )

    # Evaluate each remaining (goal, seed) pair
    pbar = tqdm(total=remaining, desc=f"agent={agent_idx} step={step//1_000_000}M",
                unit="ep", leave=False)
    for goal_idx in range(num_goals):
      goal_pos = test_goals[goal_idx]

      # Create ONE env per goal, reuse across seeds
      eval_env = _make_eval_env(
        map_name,
        goal_pos,
        device=device,
        episode_length_s=bench_cfg.episode_length_s,
      )

      for seed_idx in range(num_seeds):
        if earliest_success[agent_idx][goal_idx][seed_idx] is not None:
          continue

        env_seed = bench_cfg.test_env_seeds[seed_idx]

        torch.manual_seed(env_seed)
        np.random.seed(env_seed % (2**31))

        success = _run_episode(algo, eval_env)
        if success:
          earliest_success[agent_idx][goal_idx][seed_idx] = step
          logger.debug(
            f"  SUCCESS agent={agent_idx} goal={goal_idx} seed={seed_idx} @step {step}"
          )
        pbar.update(1)

      eval_env.close()
    pbar.close()

    dummy_env.close()

  # ── Compute metrics ────────────────────────────────────────────────────────

  # Successful runs: triples where earliest_success is not None
  sr_count = 0
  sr_min_steps: list[int] = []
  for agent_idx in range(num_agents):
    for goal_idx in range(num_goals):
      for seed_idx in range(num_seeds):
        t = earliest_success[agent_idx][goal_idx][seed_idx]
        if t is not None:
          sr_count += 1
          sr_min_steps.append(t)

  total_triples = num_agents * num_goals * num_seeds
  tsr = sr_count / total_triples if total_triples > 0 else 0.0
  lam = float(np.mean(sr_min_steps)) if sr_min_steps else float(bench_cfg.total_steps)
  ses = (bench_cfg.total_steps - lam) / bench_cfg.total_steps

  result = {
    "stage": 1,
    "map": map_name,
    "difficulty": bench_cfg.difficulty,
    "TSR": tsr,
    "SES": ses,
    "T": bench_cfg.total_steps,
    "Lambda": lam,
    "SR_count": sr_count,
    "total_triples": total_triples,
    "num_agent_seeds": num_agents,
    "num_test_goals": num_goals,
    "num_test_env_seeds": num_seeds,
    "sr_min_steps": sr_min_steps,
  }

  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / "stage1_results.json"
  with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
  logger.success(f"Stage 1 results saved: {out_path}")

  return result


def evaluate_stage2(
  bench_cfg: BenchmarkConfig,
  logs_root: Path,
  output_dir: Path,
  device: str = "cuda:0",
) -> dict[str, Any]:
  """Run Stage 2 evaluation: map generalization.

  For each held-out map, runs the same protocol as Stage 1 computed
  on policies trained on train_maps (without retraining on test maps).
  Metrics are averaged across held-out maps.
  """
  all_results: list[dict] = []

  for test_map in bench_cfg.test_maps:
    # Create a sub-benchmark config for this specific map
    sub_cfg = replace(
      bench_cfg,
      train_maps=bench_cfg.train_maps,
      test_maps=[test_map],
    )
    map_result = evaluate_stage1(sub_cfg, logs_root, output_dir / f"map_{test_map}", device)
    all_results.append(map_result)

  if not all_results:
    return {"error": "no maps evaluated"}

  avg_tsr = float(np.mean([r.get("TSR", 0.0) for r in all_results]))
  avg_ses = float(np.mean([r.get("SES", 0.0) for r in all_results]))

  summary = {
    "stage": 2,
    "difficulty": bench_cfg.difficulty,
    "train_maps": bench_cfg.train_maps,
    "test_maps": bench_cfg.test_maps,
    "TSR": avg_tsr,
    "SES": avg_ses,
    "per_map": all_results,
  }

  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / "stage2_results.json"
  with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
  logger.success(f"Stage 2 results saved: {out_path}")

  return summary


# ── CLI ──────────────────────────────────────────────────────────────────────────


def main() -> None:
  import argparse

  parser = argparse.ArgumentParser(description="HumanoidSoccerMaze Benchmark Evaluation")
  parser.add_argument("--config", type=Path, default=None,
                      help="Path to saved BenchmarkConfig JSON")
  parser.add_argument("--logs-root", type=Path, default=Path("./logs"),
                      help="Root directory containing training runs")
  parser.add_argument("--output", type=Path, default=Path("./benchmark_results"),
                      help="Output directory for results")
  parser.add_argument("--difficulty", type=str, default="easy",
                      choices=["easy", "medium", "hard"])
  parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
  parser.add_argument("--base-seed", type=int, default=0)
  parser.add_argument("--num-agent-seeds", type=int, default=3)
  parser.add_argument("--num-test-goals", type=int, default=30)
  parser.add_argument("--num-test-env-seeds", type=int, default=30)
  parser.add_argument("--total-steps", type=int, default=200_000_000)
  parser.add_argument("--eval-interval", type=int, default=5_000_000)
  parser.add_argument("--device", type=str, default="cuda:0")

  args = parser.parse_args()

  # Load or build benchmark config
  if args.config is not None:
    bench_cfg = BenchmarkConfig.load(args.config)
  else:
    bench_cfg = BenchmarkConfig(
      difficulty=args.difficulty,
      stage=args.stage,
      base_seed=args.base_seed,
      num_agent_seeds=args.num_agent_seeds,
      num_test_goals=args.num_test_goals,
      num_test_env_seeds=args.num_test_env_seeds,
      total_steps=args.total_steps,
      eval_interval=args.eval_interval,
    ).build()

  bench_cfg.save(args.output / "benchmark_config.json")

  os.environ["MUJOCO_EGL_DEVICE_ID"] = args.device.split(":")[-1]

  if bench_cfg.stage == 1:
    result = evaluate_stage1(bench_cfg, args.logs_root, args.output, device=args.device)
  else:
    result = evaluate_stage2(bench_cfg, args.logs_root, args.output, device=args.device)

  tsr = result.get("TSR", "N/A")
  ses = result.get("SES", "N/A")
  print(f"\n{'='*60}")
  print(f"  Stage {bench_cfg.stage}  |  Difficulty: {bench_cfg.difficulty}")
  print(f"  TSR = {tsr:.4f}" if isinstance(tsr, float) else f"  TSR = {tsr}")
  print(f"  SES = {ses:.4f}" if isinstance(ses, float) else f"  SES = {ses}")
  print(f"{'='*60}\n")


if __name__ == "__main__":
  main()
