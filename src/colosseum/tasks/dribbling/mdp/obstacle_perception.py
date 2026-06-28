"""Detector-like nearest-obstacle estimate (sim model of the on-robot pipeline).

Mirrors BallPerceptionModel: range-scaled position noise, lagged velocity,
detection dropout (out-of-range / random miss -> coast on constant velocity),
fixed latency, per-episode DR. Output is body-frame and carries a present bit so
the policy can distinguish 'no obstacle' from 'obstacle at the origin'.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import ObservationTermCfg

from colosseum.tasks.dribbling.mdp.obstacle_commands import ObstacleCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_NEUTRAL: float = 0.0


def _yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
  return torch.atan2(
    2.0 * (quat_w[:, 0] * quat_w[:, 3] + quat_w[:, 1] * quat_w[:, 2]),
    1.0 - 2.0 * (quat_w[:, 2] ** 2 + quat_w[:, 3] ** 2),
  )


def obstacle_state_gt(
  env: ManagerBasedRlEnv, command_name: str = "adversary"
) -> torch.Tensor:
  """Clean GT body-frame nearest-obstacle state (N, 5) = [present, px, py, vx, vy].

  Critic-only term (privileged, never deployed)."""
  model = _ground_truth(env, command_name)
  return model


class ObstaclePerceptionModel:
  """Noisy, lagged, droppable body-frame nearest-obstacle estimate. Returns (N, 5)."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    p = cfg.params
    self._command_name = str(p.get("command_name", "adversary"))

    self._sigma_pos_base = float(p.get("sigma_pos_base", 0.03))
    self._sigma_pos_per_m = float(p.get("sigma_pos_per_m", 0.05))
    self._sigma_pos_scale_range = tuple(p.get("sigma_pos_scale_range", (0.5, 2.0)))
    self._sigma_vel_range = tuple(p.get("sigma_vel_range", (0.1, 0.4)))
    self._vel_filter_alpha = float(p.get("vel_filter_alpha", 0.3))
    self._coast_vel_decay = float(p.get("coast_vel_decay", 0.98))
    self._p_miss_range = tuple(p.get("p_miss_range", (0.02, 0.2)))
    self._max_range = float(p.get("max_range", 6.0))
    self._latency = int(p.get("latency_steps", 2))

    self._dt = env.step_dt
    n, dev = env.num_envs, env.device

    self._est_pos_w = torch.zeros(n, 2, device=dev)
    self._est_vel_w = torch.zeros(n, 2, device=dev)

    self._buf = torch.zeros(self._latency + 1, n, 5, device=dev)
    self._buf_ptr = 0

    self._sigma_pos_scale = torch.ones(n, device=dev)
    self._sigma_vel = torch.full((n,), self._sigma_vel_range[0], device=dev)
    self._p_miss = torch.full((n,), self._p_miss_range[0], device=dev)

    self.reset(None)

  def _nearest_world(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = self._env.scene["robot"]
    term: ObstacleCommand = self._env.command_manager.get_term(self._command_name)
    n, dev = self._env.num_envs, self._env.device
    robot_xy = robot.data.root_link_pos_w[:, :2]

    if term.cfg.num_active == 0:
      return (
        torch.zeros(n, device=dev),
        robot_xy.clone(),
        torch.zeros(n, 2, device=dev),
      )

    obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]
    obs_vel = term.obstacle_velocities_w[:, : term.cfg.num_active]
    dist = (obs_xy - robot_xy.unsqueeze(1)).norm(dim=-1)
    min_dist, idx = dist.min(dim=-1)
    b = torch.arange(n, device=dev)
    nearest_xy = obs_xy[b, idx]
    nearest_vel = obs_vel[b, idx]
    present = (min_dist < self._max_range).float()
    return present, nearest_xy, nearest_vel

  def _to_body(
    self,
    present: torch.Tensor,
    pos_w: torch.Tensor,
    vel_w: torch.Tensor,
  ) -> torch.Tensor:
    robot = self._env.scene["robot"]
    robot_xy = robot.data.root_link_pos_w[:, :2]
    yaw = _yaw_from_quat(robot.data.root_link_quat_w)
    c, s = torch.cos(-yaw), torch.sin(-yaw)

    rel_x = pos_w[:, 0] - robot_xy[:, 0]
    rel_y = pos_w[:, 1] - robot_xy[:, 1]
    px = c * rel_x - s * rel_y
    py = s * rel_x + c * rel_y
    vx = c * vel_w[:, 0] - s * vel_w[:, 1]
    vy = s * vel_w[:, 0] + c * vel_w[:, 1]

    px = present * px + (1.0 - present) * _NEUTRAL
    py = present * py + (1.0 - present) * _NEUTRAL
    vx = present * vx + (1.0 - present) * _NEUTRAL
    vy = present * vy + (1.0 - present) * _NEUTRAL
    return torch.stack([present, px, py, vx, vy], dim=-1)

  def _sample_dr(self, env_ids: torch.Tensor) -> None:
    n, dev = len(env_ids), self._env.device

    def u(lo: float, hi: float) -> torch.Tensor:
      return torch.rand(n, device=dev) * (hi - lo) + lo

    self._sigma_pos_scale[env_ids] = u(*self._sigma_pos_scale_range)
    self._sigma_vel[env_ids] = u(*self._sigma_vel_range)
    self._p_miss[env_ids] = u(*self._p_miss_range)

  def reset(self, env_ids: torch.Tensor | None) -> dict:
    if env_ids is None:
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    present, pos_w, vel_w = self._nearest_world()
    self._est_pos_w[env_ids] = pos_w[env_ids]
    self._est_vel_w[env_ids] = vel_w[env_ids]
    world_gt = torch.cat([present.unsqueeze(-1), pos_w, vel_w], dim=-1)
    self._buf[:, env_ids, :] = world_gt[env_ids].unsqueeze(0)
    self._sample_dr(env_ids)
    return {}

  def _update(self) -> torch.Tensor:
    present, pos_w, vel_w = self._nearest_world()
    robot = self._env.scene["robot"]
    robot_xy = robot.data.root_link_pos_w[:, :2]
    n, dev = self._env.num_envs, self._env.device

    rng = (pos_w - robot_xy).norm(dim=-1)
    present_b = present.bool()
    rand_miss = torch.rand(n, device=dev) < self._p_miss
    seen = present_b & ~rand_miss
    seen_f = seen.float().unsqueeze(-1)

    sigma_pos = (
      self._sigma_pos_base + self._sigma_pos_per_m * rng
    ) * self._sigma_pos_scale
    meas_pos = pos_w + torch.randn(n, 2, device=dev) * sigma_pos.unsqueeze(-1)
    meas_vel = vel_w + torch.randn(n, 2, device=dev) * self._sigma_vel.unsqueeze(-1)

    pos_seen = meas_pos
    vel_seen = (
      1.0 - self._vel_filter_alpha
    ) * self._est_vel_w + self._vel_filter_alpha * meas_vel
    pos_unseen = self._est_pos_w + self._est_vel_w * self._dt
    vel_unseen = self._est_vel_w * self._coast_vel_decay

    self._est_pos_w = seen_f * pos_seen + (1.0 - seen_f) * pos_unseen
    self._est_vel_w = seen_f * vel_seen + (1.0 - seen_f) * vel_unseen

    world = torch.cat(
      [present.unsqueeze(-1), self._est_pos_w, self._est_vel_w], dim=-1
    )
    self._buf[self._buf_ptr] = world
    self._buf_ptr = (self._buf_ptr + 1) % (self._latency + 1)
    delayed = self._buf[self._buf_ptr]
    return self._to_body(delayed[:, 0], delayed[:, 1:3], delayed[:, 3:5])

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del env, kwargs
    return self._update()


def _ground_truth(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  robot = env.scene["robot"]
  term: ObstacleCommand = env.command_manager.get_term(command_name)
  n, dev = env.num_envs, env.device
  robot_xy = robot.data.root_link_pos_w[:, :2]
  if term.cfg.num_active == 0:
    return torch.zeros(n, 5, device=dev)
  obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]
  obs_vel = term.obstacle_velocities_w[:, : term.cfg.num_active]
  dist = (obs_xy - robot_xy.unsqueeze(1)).norm(dim=-1)
  min_dist, idx = dist.min(dim=-1)
  b = torch.arange(n, device=dev)
  yaw = _yaw_from_quat(robot.data.root_link_quat_w)
  c, s = torch.cos(-yaw), torch.sin(-yaw)
  rel_x = obs_xy[b, idx, 0] - robot_xy[:, 0]
  rel_y = obs_xy[b, idx, 1] - robot_xy[:, 1]
  px = c * rel_x - s * rel_y
  py = s * rel_x + c * rel_y
  vx = c * obs_vel[b, idx, 0] - s * obs_vel[b, idx, 1]
  vy = s * obs_vel[b, idx, 0] + c * obs_vel[b, idx, 1]
  present = (min_dist < 6.0).float()
  return torch.stack([present, px, py, vx, vy], dim=-1)
