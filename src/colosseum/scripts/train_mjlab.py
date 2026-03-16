#!/usr/bin/env python3
"""RSL-RL training script for mjlab built-in tasks.

Uses mjlab's task registry and built-in RSL-RL runner.
For colosseum tasks, use train-rsl-rl instead.

Usage:
    pixi run -e train train-mjlab Mjlab-Velocity-Flat-Unitree-G1
    pixi run -e train train-mjlab Mjlab-Velocity-Rough-Unitree-G1
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import (
  MjlabOnPolicyRunner,
  RslRlOnPolicyRunnerCfg,
  RslRlVecEnvWrapper,
)
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

# Import colosseum tasks so mjlab registry sees them too
import colosseum.tasks  # noqa: F401


@dataclass(frozen=True)
class RslRlTrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "RslRlTrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return RslRlTrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: RslRlTrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  device = "cpu" if cuda_visible == "" else f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
  seed = cfg.agent.seed

  configure_torch_backends()
  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training: device={device}, seed={seed}")
  print(f"[INFO] Log directory: {log_dir}")

  env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode=None)

  if cfg.agent.resume:
    resume_path = get_checkpoint_path(
      log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint
    )
  else:
    resume_path = None

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(cfg.agent), str(log_dir), device)

  if resume_path is not None:
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(str(resume_path))

  log_dir.mkdir(parents=True, exist_ok=True)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

  runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
  env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    RslRlTrainConfig,
    args=remaining_args,
    default=RslRlTrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )

  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  if args.gpu_ids is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    if isinstance(args.gpu_ids, list):
      os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
  os.environ["MUJOCO_GL"] = "egl"

  run_train(chosen_task, args, log_dir)


if __name__ == "__main__":
  main()
