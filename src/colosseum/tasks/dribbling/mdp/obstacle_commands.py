"""Command term: obstacle positions for adversarial dribbling.

ObstacleCommand holds the world-frame XY position of every obstacle and is
the single source of truth for where obstacles live.  It is responsible for:

  1. Sampling new positions around the environment origin at every episode
     reset (``_resample_command``).
  2. Writing those positions to the MuJoCo mocap bodies so the collision
     geometry matches (``write_mocap_pose_to_sim``).
  3. Stepping moving obstacles each environment tick (``_update_command``),
     enabled in later curriculum stages via ``max_speed > 0``.

Inactive obstacles (``k >= num_active``) are parked underground so they
have no physical effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Underground z used to park inactive obstacles out of reach.
_PARK_Z: float = -10.0


class ObstacleCommand(CommandTerm):
  """World-frame obstacle position command.

  At each episode reset, active obstacles are placed along the ball velocity
  command direction at a random distance from the env origin, so the obstacle
  starts directly in the robot's dribbling path.  Inactive obstacles are
  teleported underground.

  The command tensor exposed to the manager is the flat (N, num_obstacles*2)
  array of world-frame XY positions; observation terms read it via
  ``obstacle_positions_w``.
  """

  cfg: ObstacleCommandCfg

  def __init__(self, cfg: ObstacleCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    N = env.num_envs
    K = cfg.num_obstacles
    # World-frame XY positions, shape (N, K, 2).
    self._positions_w = torch.zeros((N, K, 2), device=env.device)
    # Last seen ball velocity direction, used to detect resamples of ball_vel.
    # Shape (N, 2); initialised to NaN so the first step always triggers a resample.
    self._last_cmd_dir = torch.full((N, 2), float("nan"), device=env.device)

  # ------------------------------------------------------------------
  # CommandTerm interface
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Flat world-frame XY positions, shape (N, num_obstacles*2)."""
    return self._positions_w.flatten(1)

  @property
  def obstacle_positions_w(self) -> torch.Tensor:
    """World-frame XY positions, shape (N, num_obstacles, 2)."""
    return self._positions_w

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    device = self._env.device
    n = len(env_ids)
    K = self.cfg.num_obstacles
    num_active = self.cfg.num_active
    d_lo, d_hi = self.cfg.distance_range

    # Invalidate stored direction so the next _update_command tick resamples.
    self._last_cmd_dir[env_ids] = float("nan")

    # Robot position is valid at reset time (events run before commands).
    robot = self._env.scene["robot"]
    robot_xy = robot.data.root_link_pos_w[env_ids, :2]  # (n, 2)

    ball_vel_cmd = self._env.command_manager.get_command("ball_vel")[env_ids, :2]  # (n, 2)
    cmd_speed = ball_vel_cmd.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    cmd_dir = ball_vel_cmd / cmd_speed  # (n, 2)

    for k in range(K):
      if k < num_active:
        distances = torch.rand(n, device=device) * (d_hi - d_lo) + d_lo
        self._positions_w[env_ids, k, 0] = robot_xy[:, 0] + distances * cmd_dir[:, 0]
        self._positions_w[env_ids, k, 1] = robot_xy[:, 1] + distances * cmd_dir[:, 1]
        self._write_obstacle_to_sim(k, env_ids, z=0.0)
      else:
        # Park underground at env origin.
        env_origin_xy = self._env.scene.env_origins[env_ids, :2]
        self._positions_w[env_ids, k, 0] = env_origin_xy[:, 0]
        self._positions_w[env_ids, k, 1] = env_origin_xy[:, 1]
        self._write_obstacle_to_sim(k, env_ids, z=_PARK_Z)

  def _update_command(self) -> None:
    """Reposition active obstacles every env tick.

    On a ball-velocity direction change the obstacle is teleported to
    ``robot_xy + distance * cmd_dir`` (using the robot's *current* world-frame
    position as the anchor so the obstacle lands in front of the robot wherever
    it has moved to).  Between resamples the obstacle drifts toward the robot
    at ``max_speed`` in world-frame along ``-cmd_dir``.
    """
    if self.cfg.num_active == 0:
      return

    device = self._env.device
    N = self._env.num_envs
    all_ids = torch.arange(N, device=device)
    dt = self._env.step_dt
    d_lo, d_hi = self.cfg.distance_range

    ball_vel_cmd = self._env.command_manager.get_command("ball_vel")[:, :2]  # (N, 2)
    cmd_speed = ball_vel_cmd.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    cmd_dir = ball_vel_cmd / cmd_speed  # (N, 2)

    robot = self._env.scene["robot"]
    robot_xy = robot.data.root_link_pos_w[:, :2]  # (N, 2)

    # Detect ball_vel direction change (dot < 0.95 ≈ ≥18° turn).
    # NaN < 0.95 is False in PyTorch, so ~(NaN >= 0.95) = True — the first
    # tick always resamples because _last_cmd_dir is initialised to NaN.
    dot = (cmd_dir * self._last_cmd_dir).sum(dim=-1)  # (N,)
    dir_changed = ~(dot >= 0.95)

    for k in range(self.cfg.num_active):
      # Teleport obstacle when direction changed; anchor to robot's current pos.
      if dir_changed.any():
        ids = dir_changed.nonzero(as_tuple=False).flatten()
        distances = torch.rand(len(ids), device=device) * (d_hi - d_lo) + d_lo
        self._positions_w[ids, k, 0] = robot_xy[ids, 0] + distances * cmd_dir[ids, 0]
        self._positions_w[ids, k, 1] = robot_xy[ids, 1] + distances * cmd_dir[ids, 1]

      # Approach: move in world-frame opposite to the dribble direction.
      if self.cfg.max_speed > 0:
        self._positions_w[:, k] -= cmd_dir * self.cfg.max_speed * dt

      self._write_obstacle_to_sim(k, all_ids, z=0.0)

    self._last_cmd_dir = cmd_dir

  def _update_metrics(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs or self.cfg.num_active == 0:
      return
    for k in range(self.cfg.num_active):
      pos_xy = self._positions_w[batch, k]
      pos_3d = torch.cat([pos_xy, torch.zeros(1, device=pos_xy.device)])
      visualizer.add_sphere(
        center=pos_3d.cpu().numpy(),
        radius=0.2,
        color=(0.9, 0.3, 0.1, 0.8),
        label=f"obs_{k}",
      )

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _write_obstacle_to_sim(
    self,
    obstacle_idx: int,
    env_ids: torch.Tensor,
    z: float,
  ) -> None:
    """Write the mocap pose for obstacle *obstacle_idx* for *env_ids*."""
    entity = self._env.scene[f"obstacle_{obstacle_idx}"]
    n = len(env_ids)
    device = self._env.device
    mocap_pose = torch.zeros((n, 7), device=device)
    mocap_pose[:, 0] = self._positions_w[env_ids, obstacle_idx, 0]
    mocap_pose[:, 1] = self._positions_w[env_ids, obstacle_idx, 1]
    mocap_pose[:, 2] = z
    mocap_pose[:, 3] = 1.0  # quaternion w (identity rotation)
    entity.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)


@dataclass(kw_only=True)
class ObstacleCommandCfg(CommandTermCfg):
  """Configuration for ObstacleCommand."""

  class_type: type[CommandTerm] = ObstacleCommand

  # Very large so obstacles only resample on episode reset, not mid-episode.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  # Total number of obstacle entities in the scene.
  num_obstacles: int = 2

  # Active obstacles at this curriculum stage (0 = none, ramps up).
  num_active: int = 0

  # Sampling range for initial distance from the env origin (metres).
  distance_range: tuple[float, float] = (2.5, 4.0)

  # Maximum approach speed in m/s (0 = static obstacles).
  max_speed: float = 0.0

  def build(self, env: ManagerBasedRlEnv) -> ObstacleCommand:
    return ObstacleCommand(self, env)
