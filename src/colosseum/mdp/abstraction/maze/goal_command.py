from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from loguru import logger
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from colosseum.mdp.observations import agent_pos
from colosseum.tasks.maze.terrain import MazeTerrainEntity

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class MazeGoalCommand(CommandTerm):
  """Command term that provides the (x, y) world position of the maze goal."""

  cfg: MazeGoalCommandCfg

  def __init__(self, cfg: MazeGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.env = env

    if self.cfg.goals is not None:
      self.valid_goal_positions = self.cfg.goals.to(device=env.device)
    else:
      assert isinstance(env.scene.terrain, MazeTerrainEntity)
      self.valid_goal_positions = env.scene.terrain.valid_goal_positions_local

    self.goal_position = torch.zeros((env.num_envs, 2), device=env.device)

    self.metrics["average_goal_distance"] = torch.zeros(env.num_envs, device=env.device)
    self.current_goal_distance = torch.zeros(env.num_envs, device=env.device)

    self.goals_initialized = False

    all_env_ids = torch.arange(self.num_envs, device=self.device)
    self._resample(all_env_ids)
    self.goals_initialized = True

  @property
  def command(self) -> torch.Tensor:
    return self.goal_position

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.static_goals and self.goals_initialized:
      return

    n = len(env_ids)
    indices = torch.randint(
      0, len(self.valid_goal_positions), (n,), device=self.device
    )
    new_goals_local = self.valid_goal_positions[indices]  # [n, 2]
    env_origins = self.env.scene.env_origins[env_ids, :2]  # [n, 2]
    self.goal_position[env_ids] = new_goals_local + env_origins

  def _update_command(self) -> None:
    if "log" not in self.env.extras:
      self.env.extras["log"] = {}
    mean_distance = torch.mean(self.current_goal_distance).item()
    self.env.extras["log"]["Metrics/goal/goal_distance"] = mean_distance

  def _update_metrics(self) -> None:
    agent_position = agent_pos(
      self.env, SceneEntityCfg("robot", site_names=("root_site",))
    )  # [num_envs, 2]
    distance = torch.norm(agent_position - self.goal_position, dim=1)
    self.metrics["average_goal_distance"] = distance
    self.current_goal_distance = distance

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    goal_pos_w = self.goal_position[batch]
    goal_pos_3d = torch.cat(
      [goal_pos_w, torch.tensor([0.05], device=goal_pos_w.device)]
    ).cpu().numpy()
    visualizer.add_sphere(
      center=goal_pos_3d,
      radius=0.03,
      color=(0.8, 0.2, 0.8, 0.5),
      label="goal",
    )


@dataclass(kw_only=True)
class MazeGoalCommandCfg(CommandTermCfg):
  """Configuration for MazeGoalCommand.

  Attributes:
      static_goals: If True, each environment gets a random goal at startup
                   and keeps the same goal for the entire training run.
                   Useful when abstraction computations (e.g. Dijkstra) are
                   expensive and depend on goal position. Default True.
      goals: Optional [K, 2] tensor of local-frame goal positions to use
             instead of reading 'g' cells from the terrain. When set,
             goals are randomly sampled from this list at each reset
             (or fixed for all envs if len(goals) == 1).
  """

  class_type: type[CommandTerm] = MazeGoalCommand
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  debug_vis: bool = True
  static_goals: bool = True
  goals: torch.Tensor | None = None

  def build(self, env: ManagerBasedRlEnv) -> MazeGoalCommand:
    return MazeGoalCommand(self, env)
