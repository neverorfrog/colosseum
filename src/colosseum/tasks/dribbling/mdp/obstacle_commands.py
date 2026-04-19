"""Command term: obstacle positions and velocities for adversarial dribbling.

ObstacleCommand holds the world-frame XY position and velocity of every
obstacle.  It is the single source of truth for where obstacles live and how
fast they are moving.  It is responsible for:

  1. Sampling new positions around the robot at every episode reset
     (``_resample_command``).
  2. Writing those positions to the MuJoCo mocap bodies so the collision
     geometry matches (``_write_obstacle_to_sim``).
  3. Stepping moving obstacles each environment tick (``_update_command``),
     enabled in later curriculum stages via ``max_speed > 0``.

Each active obstacle gets its own approach direction sampled at reset (or on
ball-velocity direction change).  The approach direction is a unit vector
pointing FROM the robot TOWARD the initial obstacle position, drawn from a
random angle relative to the commanded ball-velocity direction within
``approach_angle_range``.  The obstacle then drifts in the opposite direction
(toward the robot) at ``max_speed``.

Inactive obstacles (``k >= num_active``) are parked at ground level (z=0)
but at a large world-frame XY offset (_PARK_FAR) so they appear very far away
in any body-frame observation and register near-zero danger regardless of the
gate function used.  Parking underground is avoided because it corrupts the
viewer.

Velocity is set analytically (not finite-differenced), so it is exact and
noise-free.  Velocity is zeroed on teleport events (episode reset or
ball-velocity direction change) to avoid spurious high-danger readings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Large world-frame XY offset for inactive obstacles so they appear
# safely far away in body-frame observations (>> any detection_range).
# Obstacles are kept at z=0 (ground level) to avoid viewer corruption.
_PARK_FAR: float = 1000.0


class ObstacleCommand(CommandTerm):
  """World-frame obstacle position+velocity command.

  At each episode reset, active obstacles are placed at a random distance from
  the robot along a randomly sampled approach direction.  The approach direction
  is drawn from a uniform angle offset relative to the commanded ball-velocity
  direction, within ``cfg.approach_angle_range``.

  The command tensor is the flat (N, num_obstacles*2) array of world-frame
  XY positions.  Observation terms read positions via ``obstacle_positions_w``
  and velocities via ``obstacle_velocities_w``.
  """

  cfg: ObstacleCommandCfg

  def __init__(self, cfg: ObstacleCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    N = env.num_envs
    K = cfg.num_obstacles
    # World-frame XY positions, shape (N, K, 2).
    self._positions_w = torch.zeros((N, K, 2), device=env.device)
    # World-frame XY velocities, shape (N, K, 2).
    # Set analytically; zeroed on teleport.
    self._velocities_w = torch.zeros((N, K, 2), device=env.device)
    # Per-obstacle approach unit vectors: obstacle drifts in -approach_dir.
    # Shape (N, K, 2); initialized to forward direction.
    self._approach_dirs = torch.zeros((N, K, 2), device=env.device)
    self._approach_dirs[:, :, 0] = 1.0  # default: forward (+x)
    # Last seen ball velocity direction; NaN forces resample on first tick.
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

  @property
  def obstacle_velocities_w(self) -> torch.Tensor:
    """World-frame XY velocities, shape (N, num_obstacles, 2)."""
    return self._velocities_w

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    device = self._env.device
    n = len(env_ids)
    K = self.cfg.num_obstacles
    num_active = self.cfg.num_active
    d_lo, d_hi = self.cfg.distance_range

    # Invalidate stored direction so the next _update_command tick resamples.
    self._last_cmd_dir[env_ids] = float("nan")

    # Zero all velocities at episode start (teleport, not drift).
    self._velocities_w[env_ids] = 0.0

    # Robot position is valid at reset time (events run before commands).
    robot = self._env.scene["robot"]
    robot_xy = robot.data.root_link_pos_w[env_ids, :2]  # (n, 2)

    ball_vel_cmd = self._env.command_manager.get_command("ball_vel")[env_ids, :2]
    cmd_speed = ball_vel_cmd.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    cmd_dir = ball_vel_cmd / cmd_speed  # (n, 2)
    base_angle = torch.atan2(cmd_dir[:, 1], cmd_dir[:, 0])  # (n,)

    lo, hi = self.cfg.approach_angle_range

    for k in range(K):
      if k < num_active:
        # Sample random approach direction relative to ball-velocity command.
        offsets = torch.rand(n, device=device) * (hi - lo) + lo
        angle_k = base_angle + offsets
        approach_dir = torch.stack([torch.cos(angle_k), torch.sin(angle_k)], dim=-1)
        self._approach_dirs[env_ids, k] = approach_dir

        distances = torch.rand(n, device=device) * (d_hi - d_lo) + d_lo
        self._positions_w[env_ids, k] = robot_xy + distances.unsqueeze(-1) * approach_dir
        self._write_obstacle_to_sim(k, env_ids, z=0.0)
      else:
        # Park far away at ground level so the body-frame distance is always
        # >> any detection_range, without going underground (viewer-safe).
        env_origin_xy = self._env.scene.env_origins[env_ids, :2]
        self._positions_w[env_ids, k, 0] = env_origin_xy[:, 0] + _PARK_FAR
        self._positions_w[env_ids, k, 1] = env_origin_xy[:, 1] + _PARK_FAR
        self._write_obstacle_to_sim(k, env_ids, z=0.0)

  def _update_command(self) -> None:
    """Reposition active obstacles every env tick.

    On a ball-velocity direction change the obstacle is teleported to a fresh
    random position and its approach direction re-sampled.  Between resamples
    the obstacle drifts toward the robot at ``max_speed`` along its stored
    approach direction (``-approach_dir``).
    """
    if self.cfg.num_active == 0:
      return

    device = self._env.device
    N = self._env.num_envs
    all_ids = torch.arange(N, device=device)
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

    lo, hi = self.cfg.approach_angle_range
    base_angle = torch.atan2(cmd_dir[:, 1], cmd_dir[:, 0])  # (N,)

    for k in range(self.cfg.num_active):
      # --- Teleport + re-sample approach direction on ball-vel direction change ---
      if dir_changed.any():
        ids = dir_changed.nonzero(as_tuple=False).flatten()
        n = len(ids)
        offsets = torch.rand(n, device=device) * (hi - lo) + lo
        angle_k = base_angle[ids] + offsets
        approach_dir = torch.stack([torch.cos(angle_k), torch.sin(angle_k)], dim=-1)
        self._approach_dirs[ids, k] = approach_dir

        distances = torch.rand(n, device=device) * (d_hi - d_lo) + d_lo
        self._positions_w[ids, k] = robot_xy[ids] + distances.unsqueeze(-1) * approach_dir
        # Teleport → zero velocity.
        self._velocities_w[ids, k] = 0.0

      # --- Continuous drift along stored approach direction ---
      if self.cfg.max_speed > 0:
        drifting = ~dir_changed  # (N,)
        dt = self._env.step_dt
        approach_k = self._approach_dirs[:, k]  # (N, 2)
        self._positions_w[:, k] -= approach_k * self.cfg.max_speed * dt
        self._velocities_w[drifting, k] = -approach_k[drifting] * self.cfg.max_speed
      else:
        non_teleported = ~dir_changed
        self._velocities_w[non_teleported, k] = 0.0

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

  # Sampling range for initial distance from the robot (metres).
  distance_range: tuple[float, float] = (2.5, 4.0)

  # Maximum approach speed in m/s (0 = static obstacles).
  max_speed: float = 0.0

  # Angular offset range (radians) relative to ball-velocity direction for
  # sampling each obstacle's approach direction.  (-pi, pi) = fully random;
  # (-pi/2, pi/2) = frontal half-space only.
  approach_angle_range: tuple[float, float] = (-math.pi, math.pi)

  def build(self, env: ManagerBasedRlEnv) -> ObstacleCommand:
    return ObstacleCommand(self, env)
