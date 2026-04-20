"""Command term: obstacle positions and velocities for adversarial dribbling.

ObstacleCommand controls obstacle placement and motion in the simulator.
The obstacles are curriculum-driven and can switch between several behaviors:

  - ``none``            : no active obstacle
  - ``static_blocker``  : static blocker in front of the commanded ball path
  - ``lateral_blocker`` : blocker near the ball path with lateral motion
  - ``ball_attacker``   : obstacle moves toward the ball with smoothly changing
                          random speed and heading bias
  - ``mixed_attackers`` : three-obstacle scene with one ball attacker, one
                          blocker, and one distractor

Positions and velocities are maintained in world frame, then written to the
MuJoCo mocap bodies every step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Large world-frame XY offset for inactive obstacles so they appear
# safely far away in body-frame observations (>> any detection_range).
# Obstacles are kept at z=0 (ground level) to avoid viewer corruption.
_PARK_FAR: float = 1000.0


class ObstacleCommand(CommandTerm):
  """World-frame obstacle position+velocity command."""

  cfg: ObstacleCommandCfg

  def __init__(self, cfg: ObstacleCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    N = env.num_envs
    K = cfg.num_obstacles
    # World-frame XY positions, shape (N, K, 2).
    self._positions_w = torch.zeros((N, K, 2), device=env.device)
    # World-frame XY velocities, shape (N, K, 2).
    self._velocities_w = torch.zeros((N, K, 2), device=env.device)
    # Motion state for smooth stochastic obstacle behavior.
    self._speed_targets = torch.zeros((N, K), device=env.device)
    self._lateral_signs = torch.ones((N, K), device=env.device)
    self._tangent_mix = torch.zeros((N, K), device=env.device)
    self._random_dirs = torch.zeros((N, K, 2), device=env.device)
    self._random_dirs[:, :, 0] = 1.0
    self._resample_timers = torch.zeros((N, K), device=env.device)

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
    K = self.cfg.num_obstacles
    num_active = self.cfg.num_active
    self._velocities_w[env_ids] = 0.0
    self._speed_targets[env_ids] = 0.0
    self._tangent_mix[env_ids] = 0.0
    self._lateral_signs[env_ids] = 1.0
    self._resample_timers[env_ids] = 0.0

    for k in range(K):
      if k < num_active:
        self._positions_w[env_ids, k] = self._sample_spawn_positions(env_ids, k)
        self._resample_motion_state(env_ids, k)
        vel_target = self._compute_velocity_target(env_ids, k)
        self._velocities_w[env_ids, k] = vel_target
        self._write_obstacle_to_sim(k, env_ids, z=0.0)
      else:
        self._park_obstacle(k, env_ids)

  def _update_command(self) -> None:
    device = self._env.device
    N = self._env.num_envs
    all_ids = torch.arange(N, device=device)
    dt = self._env.step_dt

    for k in range(self.cfg.num_obstacles):
      if k >= self.cfg.num_active:
        self._park_obstacle(k, all_ids)
        continue

      parked = self._is_parked(k)
      if parked.any():
        ids = parked.nonzero(as_tuple=False).flatten()
        self._positions_w[ids, k] = self._sample_spawn_positions(ids, k)
        self._resample_motion_state(ids, k)
        self._velocities_w[ids, k] = self._compute_velocity_target(ids, k)

      needs_respawn = self._needs_respawn(all_ids, k)
      if needs_respawn.any():
        ids = needs_respawn.nonzero(as_tuple=False).flatten()
        self._positions_w[ids, k] = self._sample_spawn_positions(ids, k)
        self._resample_motion_state(ids, k)
        self._velocities_w[ids, k] = self._compute_velocity_target(ids, k)

      self._resample_timers[:, k] -= dt
      expired = self._resample_timers[:, k] <= 0.0
      if expired.any():
        ids = expired.nonzero(as_tuple=False).flatten()
        self._resample_motion_state(ids, k)

      vel_target = self._compute_velocity_target(all_ids, k)
      alpha = self.cfg.velocity_smoothing
      self._velocities_w[:, k] = (
        (1.0 - alpha) * self._velocities_w[:, k] + alpha * vel_target
      )
      self._positions_w[:, k] += self._velocities_w[:, k] * dt
      self._write_obstacle_to_sim(k, all_ids, z=0.0)

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

  def _cmd_and_side_dirs(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ball_vel_cmd = self._env.command_manager.get_command("ball_vel")[env_ids, :2]
    cmd_speed = ball_vel_cmd.norm(dim=-1, keepdim=True)
    default_dir = torch.zeros_like(ball_vel_cmd)
    default_dir[:, 0] = 1.0
    cmd_dir = torch.where(
      cmd_speed > 1e-3,
      ball_vel_cmd / cmd_speed.clamp(min=1e-3),
      default_dir,
    )
    side_dir = torch.stack([-cmd_dir[:, 1], cmd_dir[:, 0]], dim=-1)
    return cmd_dir, side_dir

  def _role_for_obstacle(self, obstacle_idx: int) -> str:
    behavior = self.cfg.behavior
    if behavior == "mixed_attackers":
      if obstacle_idx == 0:
        return "ball_attacker"
      if obstacle_idx == 1:
        return "lateral_blocker"
      return "distractor"
    return behavior

  def _sample_spawn_positions(
    self,
    env_ids: torch.Tensor,
    obstacle_idx: int,
  ) -> torch.Tensor:
    role = self._role_for_obstacle(obstacle_idx)
    n = len(env_ids)
    device = self._env.device
    ball_xy = self._env.scene["ball"].data.root_link_pos_w[env_ids, :2]
    cmd_dir, side_dir = self._cmd_and_side_dirs(env_ids)
    d_lo, d_hi = self.cfg.distance_range
    l_lo, l_hi = self.cfg.lateral_offset_range

    if role in {"static_blocker", "lateral_blocker", "ball_attacker"}:
      forward = torch.rand(n, device=device) * (d_hi - d_lo) + d_lo
      lateral = torch.rand(n, device=device) * (l_hi - l_lo) + l_lo
      return ball_xy + forward.unsqueeze(-1) * cmd_dir + lateral.unsqueeze(-1) * side_dir

    if role == "distractor":
      angles = torch.rand(n, device=device) * 2.0 * math.pi - math.pi
      radius = torch.rand(n, device=device) * (d_hi - d_lo) + d_lo
      random_dir = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
      self._random_dirs[env_ids, obstacle_idx] = random_dir
      return ball_xy + radius.unsqueeze(-1) * random_dir

    return ball_xy

  def _sample_speed(self, env_ids: torch.Tensor) -> torch.Tensor:
    lo = self.cfg.min_speed
    hi = self.cfg.max_speed
    if hi <= lo:
      return torch.full((len(env_ids),), lo, device=self._env.device)
    return torch.rand(len(env_ids), device=self._env.device) * (hi - lo) + lo

  def _sample_timer(self, env_ids: torch.Tensor) -> torch.Tensor:
    lo, hi = self.cfg.velocity_resample_time_range
    if hi <= lo:
      return torch.full((len(env_ids),), lo, device=self._env.device)
    return torch.rand(len(env_ids), device=self._env.device) * (hi - lo) + lo

  def _resample_motion_state(self, env_ids: torch.Tensor, obstacle_idx: int) -> None:
    role = self._role_for_obstacle(obstacle_idx)
    if role in {"none", "static_blocker"}:
      self._speed_targets[env_ids, obstacle_idx] = 0.0
      self._tangent_mix[env_ids, obstacle_idx] = 0.0
      self._lateral_signs[env_ids, obstacle_idx] = 1.0
      self._resample_timers[env_ids, obstacle_idx] = 1e9
      return

    self._speed_targets[env_ids, obstacle_idx] = self._sample_speed(env_ids)
    self._resample_timers[env_ids, obstacle_idx] = self._sample_timer(env_ids)

    if role == "lateral_blocker":
      sign = torch.randint(0, 2, (len(env_ids),), device=self._env.device, dtype=torch.int64)
      self._lateral_signs[env_ids, obstacle_idx] = sign.float() * 2.0 - 1.0
      self._tangent_mix[env_ids, obstacle_idx] = 0.0
    elif role == "ball_attacker":
      lo, hi = self.cfg.attack_tangent_range
      self._tangent_mix[env_ids, obstacle_idx] = (
        torch.rand(len(env_ids), device=self._env.device) * (hi - lo) + lo
      )
    elif role == "distractor":
      angles = torch.rand(len(env_ids), device=self._env.device) * 2.0 * math.pi - math.pi
      self._random_dirs[env_ids, obstacle_idx] = torch.stack(
        [torch.cos(angles), torch.sin(angles)], dim=-1
      )

  def _compute_velocity_target(
    self,
    env_ids: torch.Tensor,
    obstacle_idx: int,
  ) -> torch.Tensor:
    role = self._role_for_obstacle(obstacle_idx)
    n = len(env_ids)
    if role in {"none", "static_blocker"}:
      return torch.zeros((n, 2), device=self._env.device)

    speed = self._speed_targets[env_ids, obstacle_idx].unsqueeze(-1)
    _, side_dir = self._cmd_and_side_dirs(env_ids)

    if role == "lateral_blocker":
      sign = self._lateral_signs[env_ids, obstacle_idx].unsqueeze(-1)
      return sign * side_dir * speed

    if role == "ball_attacker":
      ball_xy = self._env.scene["ball"].data.root_link_pos_w[env_ids, :2]
      obs_xy = self._positions_w[env_ids, obstacle_idx]
      to_ball = ball_xy - obs_xy
      to_ball_dir = to_ball / to_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)
      tangent_dir = torch.stack([-to_ball_dir[:, 1], to_ball_dir[:, 0]], dim=-1)
      mix = self._tangent_mix[env_ids, obstacle_idx].unsqueeze(-1)
      attack_dir = to_ball_dir + mix * tangent_dir
      attack_dir = attack_dir / attack_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
      return attack_dir * speed

    if role == "distractor":
      return self._random_dirs[env_ids, obstacle_idx] * speed

    return torch.zeros((n, 2), device=self._env.device)

  def _is_parked(self, obstacle_idx: int) -> torch.Tensor:
    env_origin_xy = self._env.scene.env_origins[:, :2]
    offset = (self._positions_w[:, obstacle_idx] - env_origin_xy).abs().max(dim=-1).values
    return offset > (_PARK_FAR / 2.0)

  def _obstacle_pos_body(
    self,
    env_ids: torch.Tensor,
    obstacle_idx: int,
  ) -> torch.Tensor:
    robot = self._env.scene["robot"]
    quat_w = robot.data.root_link_quat_w[env_ids]
    quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
    robot_xy = robot.data.root_link_pos_w[env_ids, :2]
    rel_xy = self._positions_w[env_ids, obstacle_idx] - robot_xy
    rel_3d = torch.cat([rel_xy, torch.zeros(len(env_ids), 1, device=self._env.device)], dim=-1)
    return quat_apply(quat_conj, rel_3d)[:, :2]

  def _needs_respawn(
    self,
    env_ids: torch.Tensor,
    obstacle_idx: int,
  ) -> torch.Tensor:
    role = self._role_for_obstacle(obstacle_idx)
    if role == "none":
      return torch.zeros(len(env_ids), dtype=torch.bool, device=self._env.device)

    robot_xy = self._env.scene["robot"].data.root_link_pos_w[env_ids, :2]
    ball_xy = self._env.scene["ball"].data.root_link_pos_w[env_ids, :2]
    obs_xy = self._positions_w[env_ids, obstacle_idx]
    obs_pos_b = self._obstacle_pos_body(env_ids, obstacle_idx)

    robot_dist = (obs_xy - robot_xy).norm(dim=-1)
    ball_dist = (obs_xy - ball_xy).norm(dim=-1)
    behind = obs_pos_b[:, 0] < self.cfg.respawn_behind_x_threshold
    far_from_robot = robot_dist > self.cfg.respawn_robot_distance
    far_from_ball = ball_dist > self.cfg.respawn_ball_distance

    if role == "ball_attacker":
      return behind | far_from_ball

    return behind | far_from_robot

  def _park_obstacle(self, obstacle_idx: int, env_ids: torch.Tensor) -> None:
    env_origin_xy = self._env.scene.env_origins[env_ids, :2]
    self._positions_w[env_ids, obstacle_idx, 0] = env_origin_xy[:, 0] + _PARK_FAR
    self._positions_w[env_ids, obstacle_idx, 1] = env_origin_xy[:, 1] + _PARK_FAR
    self._velocities_w[env_ids, obstacle_idx] = 0.0
    self._speed_targets[env_ids, obstacle_idx] = 0.0
    self._resample_timers[env_ids, obstacle_idx] = 0.0
    self._write_obstacle_to_sim(obstacle_idx, env_ids, z=0.0)

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

  # Spawn distance in metres along the behavior-specific forward/radial direction.
  distance_range: tuple[float, float] = (1.5, 3.0)

  # Behavior mode for active obstacles.
  behavior: str = "none"

  # Speed range in m/s for moving obstacles.
  min_speed: float = 0.0
  max_speed: float = 0.0

  # Spawn offset in metres perpendicular to the commanded ball direction.
  lateral_offset_range: tuple[float, float] = (-0.8, 0.8)

  # Resample target speed/bias every random interval in seconds.
  velocity_resample_time_range: tuple[float, float] = (0.5, 1.0)

  # First-order smoothing toward the resampled velocity target.
  velocity_smoothing: float = 0.2

  # Tangential heading bias for ball attackers; 0 = straight to ball.
  attack_tangent_range: tuple[float, float] = (-0.35, 0.35)

  # Respawn an obstacle after the encounter is over instead of waiting
  # for the episode to terminate.
  respawn_behind_x_threshold: float = -0.2
  respawn_robot_distance: float = 4.0
  respawn_ball_distance: float = 3.0

  def build(self, env: ManagerBasedRlEnv) -> ObstacleCommand:
    return ObstacleCommand(self, env)
