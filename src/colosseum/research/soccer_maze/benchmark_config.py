"""HumanoidSoccerMaze benchmark configuration.

Defines benchmark parameters, goal-splitting logic, and seed generation
following the protocol described in the paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from colosseum.tasks.maze.maps import MAPS, TEST_MAPS
from colosseum.tasks.maze.maze import Maze, MazeCfg
from colosseum.utils.grid_frame import GridFrame

_EASY = "umaze"
_MEDIUM = "medium"
_HARD = "large"

DIFFICULTY_MAPS: dict[str, list[str]] = {
  "easy": [_EASY],
  "medium": [_MEDIUM],
  "hard": [_HARD],
}


def _free_cells_local(maze_map: list[list], cell_size: float) -> torch.Tensor:
  """Return [K, 2] tensor of local (x, y) centres of all free (non-wall) cells."""
  maze = Maze(MazeCfg(maze_map=maze_map, cell_size=cell_size))
  return torch.tensor(maze.valid_free_positions_local, dtype=torch.float32)


def _split_train_test(
  cells: torch.Tensor,
  num_train: int,
  num_test: int,
  rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
  """Randomly split free cells into disjoint train/test goal sets.

  Caps requested counts at floor(K/2) each when the map has
  insufficient free cells.  Returns (train_goals, test_goals,
  actual_train, actual_test).
  """
  K = cells.shape[0]
  max_per_set = K // 2
  actual_train = min(num_train, max_per_set)
  actual_test = min(num_test, max_per_set)
  needed = actual_train + actual_test

  indices = rng.permutation(K)[:needed]
  shuffled = cells[indices]
  return shuffled[:actual_train], shuffled[actual_train:], actual_train, actual_test


@dataclass
class BenchmarkConfig:
  """Immutable configuration for one HumanoidSoccerMaze benchmark run.

  Attributes:
      difficulty: "easy", "medium", or "hard".
      num_agent_seeds: |S^ag| — number of independent agent seeds.
      num_train_goals: |G_train| — goals seen during training.
      num_test_goals: |G_test| — held-out goals for evaluation.
      num_train_env_seeds: |S^env_train| — env seeds used during training.
      num_test_env_seeds: |S^env_test| — env seeds used during evaluation.
      total_steps: T — total training environment steps.
      eval_interval: E — checkpoint interval in env steps.
      episode_length_s: Episode timeout in seconds.
      num_envs: Number of parallel environments for training.
      ppo_steps_per_env: PPO rollout horizon.
      base_seed: Seed used to derive all other seeds deterministically.
      stage: 1 (goal generalization) or 2 (map generalization).
  """

  difficulty: str = "easy"
  num_agent_seeds: int = 3
  num_train_goals: int = 30
  num_test_goals: int = 30
  num_train_env_seeds: int = 30
  num_test_env_seeds: int = 30
  total_steps: int = 200_000_000
  eval_interval: int = 5_000_000
  episode_length_s: float = 120.0
  num_envs: int = 2048
  ppo_steps_per_env: int = 24
  base_seed: int = 0
  stage: int = 1

  # Derived after build()
  train_maps: list[str] = field(default_factory=list)
  test_maps: list[str] = field(default_factory=list)
  agent_seeds: list[int] = field(default_factory=list)
  train_env_seeds: list[int] = field(default_factory=list)
  test_env_seeds: list[int] = field(default_factory=list)
  _train_goals_by_map: dict[str, torch.Tensor] = field(default_factory=dict)
  _test_goals_by_map: dict[str, torch.Tensor] = field(default_factory=dict)

  def build(self, rng: np.random.Generator | None = None) -> BenchmarkConfig:
    """Derive seed and goal splits from base_seed and difficulty settings."""
    if rng is None:
      rng = np.random.default_rng(self.base_seed)

    base_maps = DIFFICULTY_MAPS[self.difficulty]

    if self.stage == 1:
      train_maps = [base_maps[0]]
      test_maps = [base_maps[0]]
    else:
      train_maps = base_maps[: len(base_maps) // 2]
      test_maps = base_maps[len(base_maps) // 2 :]
      if len(train_maps) == 0 or len(test_maps) == 0:
        raise ValueError(
          f"Stage 2 needs at least 2 maps in difficulty '{self.difficulty}', "
          f"got {len(base_maps)}"
        )

    # Seeds
    seed_stream = [int(rng.integers(0, 2**31)) for _ in range(3)]
    agent_seeds = list(range(self.base_seed, self.base_seed + self.num_agent_seeds))
    train_env_seeds = [self.base_seed + 1000 + i for i in range(self.num_train_env_seeds)]
    test_env_seeds = [self.base_seed + 2000 + i for i in range(self.num_test_env_seeds)]

    # Goal splits — capped per map when insufficient free cells
    train_goals_by_map: dict[str, torch.Tensor] = {}
    test_goals_by_map: dict[str, torch.Tensor] = {}
    actual_train = self.num_train_goals
    actual_test = self.num_test_goals
    for map_name in set(train_maps + test_maps):
      m = MAPS[map_name]
      cells = _free_cells_local(m, cell_size=1.1)
      K = cells.shape[0]
      tg, vg, at, vt = _split_train_test(cells, self.num_train_goals, self.num_test_goals, rng)
      train_goals_by_map[map_name] = tg
      test_goals_by_map[map_name] = vg
      actual_train = min(actual_train, at)
      actual_test = min(actual_test, vt)
      if at < self.num_train_goals or vt < self.num_test_goals:
        from loguru import logger as _logger
        _logger.warning(
          f"Map '{map_name}' has only {K} free cells. "
          f"Capping goals: train {self.num_train_goals}→{at}, test {self.num_test_goals}→{vt}"
        )

    return BenchmarkConfig(
      difficulty=self.difficulty,
      num_agent_seeds=self.num_agent_seeds,
      num_train_goals=actual_train,
      num_test_goals=actual_test,
      num_train_env_seeds=self.num_train_env_seeds,
      num_test_env_seeds=self.num_test_env_seeds,
      total_steps=self.total_steps,
      eval_interval=self.eval_interval,
      episode_length_s=self.episode_length_s,
      num_envs=self.num_envs,
      ppo_steps_per_env=self.ppo_steps_per_env,
      base_seed=self.base_seed,
      stage=self.stage,
      train_maps=train_maps,
      test_maps=test_maps,
      agent_seeds=agent_seeds,
      train_env_seeds=train_env_seeds,
      test_env_seeds=test_env_seeds,
      _train_goals_by_map=train_goals_by_map,
      _test_goals_by_map=test_goals_by_map,
    )

  def train_goals(self, map_name: str) -> torch.Tensor:
    return self._train_goals_by_map[map_name]

  def test_goals(self, map_name: str) -> torch.Tensor:
    return self._test_goals_by_map[map_name]

  @property
  def train_map(self) -> str:
    return self.train_maps[0]

  @property
  def test_map(self) -> str:
    return self.test_maps[0]

  def checkpoint_steps(self) -> list[int]:
    """Return the list of global_step values at which checkpoints are saved."""
    steps: list[int] = []
    t = self.eval_interval
    while t <= self.total_steps:
      steps.append(t)
      t += self.eval_interval
    return steps

  def to_dict(self) -> dict[str, Any]:
    return {
      "difficulty": self.difficulty,
      "stage": self.stage,
      "num_agent_seeds": self.num_agent_seeds,
      "num_train_goals": self.num_train_goals,
      "num_test_goals": self.num_test_goals,
      "num_train_env_seeds": self.num_train_env_seeds,
      "num_test_env_seeds": self.num_test_env_seeds,
      "total_steps": self.total_steps,
      "eval_interval": self.eval_interval,
      "episode_length_s": self.episode_length_s,
      "num_envs": self.num_envs,
      "base_seed": self.base_seed,
      "train_maps": self.train_maps,
      "test_maps": self.test_maps,
      "agent_seeds": self.agent_seeds,
    }

  def save(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = self.to_dict()
    data["train_goals"] = {k: v.tolist() for k, v in self._train_goals_by_map.items()}
    data["test_goals"] = {k: v.tolist() for k, v in self._test_goals_by_map.items()}
    with open(path, "w") as f:
      json.dump(data, f, indent=2)

  @classmethod
  def load(cls, path: Path) -> BenchmarkConfig:
    with open(path) as f:
      data = json.load(f)
    cfg = BenchmarkConfig(**{k: v for k, v in data.items() if k not in ("train_goals", "test_goals")})
    cfg._train_goals_by_map = {
      k: torch.tensor(v, dtype=torch.float32) for k, v in data.get("train_goals", {}).items()
    }
    cfg._test_goals_by_map = {
      k: torch.tensor(v, dtype=torch.float32) for k, v in data.get("test_goals", {}).items()
    }
    return cfg


def default_easy_stage1() -> BenchmarkConfig:
  """Default benchmark config for Easy difficulty, Stage 1."""
  return BenchmarkConfig(difficulty="easy", stage=1).build()


def default_medium_stage1() -> BenchmarkConfig:
  return BenchmarkConfig(difficulty="medium", stage=1).build()


def default_hard_stage1() -> BenchmarkConfig:
  return BenchmarkConfig(difficulty="hard", stage=1).build()
