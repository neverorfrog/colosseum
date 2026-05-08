"""Training launcher for HumanoidSoccerMaze benchmark.

Runs independent training jobs for each agent seed using train goals (G_train)
and saves checkpoints at every eval_interval steps.
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
from loguru import logger
from mjlab.utils.torch import configure_torch_backends

import colosseum.tasks  # noqa: F401
from colosseum.mdp.abstraction.maze.goal_command import MazeGoalCommandCfg
from colosseum.utils.torch import get_device, set_seed

from .benchmark_config import BenchmarkConfig


def train_agent_seed(
  bench_cfg: BenchmarkConfig,
  agent_seed: int,
  map_name: str,
  logs_dir: Path,
  device: str = "cuda:0",
) -> None:
  """Train a single agent seed on a specific map with train goals.

  Args:
      bench_cfg: Fully-built benchmark config.
      agent_seed: Agent seed index (determines policy init and gradient sampling).
      map_name: Maze map name to train on.
      logs_dir: Root logs directory for this run's checkpoints.
      device: CUDA device (e.g. "cuda:0").
  """
  from colosseum.tasks.soccer_maze.config.t1_23dof import t1_soccer_maze_env_cfg

  train_goals = bench_cfg.train_goals(map_name)
  num_envs = bench_cfg.num_envs

  # Build env config with train goals
  cfg = t1_soccer_maze_env_cfg(scenario=map_name, num_envs=num_envs, play=False)
  cfg = replace(
    cfg,
    commands={
      **cfg.commands,
      "goal": MazeGoalCommandCfg(
        static_goals=True,
        goals=train_goals,
      ),
    },
    scene=replace(cfg.scene, num_envs=num_envs),
  )

  set_seed(agent_seed)
  configure_torch_backends()

  device_id = int(device.split(":")[-1])
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
  device_obj = get_device(cuda=True, device_id=device_id)

  env = cfg.class_type(cfg=cfg, device=str(device_obj))

  from colosseum.tasks.soccer_maze.config.t1_23dof.algo_cfg import (
    t1_soccer_maze_ppo_cfg,
  )
  algo_cfg = replace(t1_soccer_maze_ppo_cfg(), learning_steps=bench_cfg.total_steps)

  module_path, class_name = algo_cfg.target.rsplit(":", 1)
  module = importlib.import_module(module_path)
  algo_class = getattr(module, class_name)

  def _noop_log(_m, _s):
    pass

  algo = algo_class(
    config=algo_cfg,
    env=env,
    device=device_obj,
    log_fn=_noop_log,
    log_interval=100_000_000,  # don't log during benchmark training
  )

  run_dir = logs_dir / f"benchmark_{map_name}_agentseed{agent_seed}"
  run_dir.mkdir(parents=True, exist_ok=True)

  ckpt_dir = run_dir / "checkpoints"
  algo.configure_checkpointing(ckpt_dir, save_interval=bench_cfg.eval_interval)

  logger.info(f"[agent_seed={agent_seed}] Training on map '{map_name}'")
  logger.info(f"  Total steps: {bench_cfg.total_steps}")
  logger.info(f"  Eval interval: {bench_cfg.eval_interval}")
  logger.info(f"  Train goals: {train_goals.shape[0]}")
  logger.info(f"  Run dir: {run_dir}")

  start_time = time.time()

  try:
    algo.train()
  except KeyboardInterrupt:
    logger.warning("Training interrupted by user")
    algo.save(ckpt_dir / "interrupted.pt", global_step=algo.global_step)
    model_name = f"model_{algo.global_step:07d}.pt"
    algo.save(ckpt_dir / model_name, global_step=algo.global_step)
    logger.info(f"Saved interrupted checkpoint: {model_name}")

  elapsed = time.time() - start_time
  logger.success(
    f"[agent_seed={agent_seed}] Training complete in {elapsed:.0f}s "
    f"({elapsed / 3600:.1f}h)"
  )

  env.close()
  if device_obj.type == "cuda":
    torch.cuda.empty_cache()


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Train benchmark policies for HumanoidSoccerMaze"
  )
  parser.add_argument("--config", type=Path, default=None,
                      help="Path to saved BenchmarkConfig JSON")
  parser.add_argument("--difficulty", type=str, default="easy")
  parser.add_argument("--stage", type=int, default=1)
  parser.add_argument("--base-seed", type=int, default=0)
  parser.add_argument("--num-agent-seeds", type=int, default=3)
  parser.add_argument("--num-train-goals", type=int, default=30)
  parser.add_argument("--num-test-goals", type=int, default=30)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--total-steps", type=int, default=200_000_000)
  parser.add_argument("--eval-interval", type=int, default=5_000_000)
  parser.add_argument("--agent-seed", type=int, default=None,
                      help="Run only this agent seed (for parallel launches)")
  parser.add_argument("--logs-dir", type=Path, default=Path("./logs"))
  parser.add_argument("--device", type=str, default="cuda:0")

  args = parser.parse_args()

  if args.config is not None:
    bench_cfg = BenchmarkConfig.load(args.config)
  else:
    bench_cfg = BenchmarkConfig(
      difficulty=args.difficulty,
      stage=args.stage,
      base_seed=args.base_seed,
      num_agent_seeds=args.num_agent_seeds,
      num_train_goals=args.num_train_goals,
      num_test_goals=args.num_test_goals,
      num_envs=args.num_envs,
      total_steps=args.total_steps,
      eval_interval=args.eval_interval,
    ).build()

  map_name = bench_cfg.train_map

  if args.agent_seed is not None:
    seeds = [args.agent_seed]
  else:
    seeds = bench_cfg.agent_seeds

  for seed in seeds:
    train_agent_seed(
      bench_cfg,
      agent_seed=seed,
      map_name=map_name,
      logs_dir=args.logs_dir,
      device=args.device,
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
  main()
