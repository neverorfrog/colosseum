"""Stateful ball-state estimator that mimics the on-robot perception pipeline.

On the robot, the policy never sees the true ball state: it reads the output of
``WorldModelSimplified`` (YOLO ``RawDetections`` -> ``KalmanFilter3D`` -> ``Ball``),
which is a filtered, lagged, occasionally-stale estimate. Training on clean GT
ball observations leaves that whole gap unmodelled.

``BallPerceptionModel`` reproduces the *output* characteristics of that pipeline
on top of the simulator's GT ball state:

  * range-scaled Gaussian position noise,
  * lagged/smoothed velocity (exponential filter, like the KF velocity),
  * detection dropout (out of FOV / too close-under the body / random miss) ->
    the estimate coasts on a constant-velocity prediction while unseen,
  * fixed sensing+filtering latency (ring buffer).

The state is kept in the *world* frame and transformed to the robot body frame
with the CURRENT robot pose at output time, so a stale (coasting) estimate
correctly stays fixed in the world while the robot turns -- which matters during
the dribble-and-turn maneuver where the ball is often dropped under the body.

The per-episode noise band is domain-randomized in ``reset`` so the policy is
robust across the band rather than tuned to one operating point.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers import ObservationTermCfg

from colosseum.tasks.dribbling.mdp.observations import (
  ball_position,
  ball_velocity_xy,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
  """Yaw (rad) from a (N, 4) wxyz quaternion."""
  return torch.atan2(
    2.0 * (quat_w[:, 0] * quat_w[:, 3] + quat_w[:, 1] * quat_w[:, 2]),
    1.0 - 2.0 * (quat_w[:, 2] ** 2 + quat_w[:, 3] ** 2),
  )


def ball_state_gt(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Clean GT body-frame ball state (N, 4) = [px, py, vx, vy] for the critic."""
  return torch.cat([ball_position(env), ball_velocity_xy(env)], dim=-1)


class BallPerceptionModel:
  """Noisy, lagged, droppable body-frame ball-state estimate. Returns (N, 4)."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    p = cfg.params

    # Position noise: sigma = (base + per_m * range) * per-episode scale.
    self._sigma_pos_base = float(p.get("sigma_pos_base", 0.02))
    self._sigma_pos_per_m = float(p.get("sigma_pos_per_m", 0.03))
    self._sigma_pos_scale_range = tuple(p.get("sigma_pos_scale_range", (0.5, 1.5)))
    # Velocity measurement noise (per-episode) + lag filter coefficient.
    self._sigma_vel_range = tuple(p.get("sigma_vel_range", (0.1, 0.3)))
    self._vel_filter_alpha = float(p.get("vel_filter_alpha", 0.3))
    # Coast: per-step velocity decay applied while the ball is unseen.
    self._coast_vel_decay = float(p.get("coast_vel_decay", 0.98))
    # Dropout drivers.
    self._p_miss_range = tuple(p.get("p_miss_range", (0.02, 0.15)))
    self._close_range = float(p.get("close_range_dropout", 0.3))
    self._max_range = float(p.get("max_range", 6.0))
    self._fov_half = float(p.get("fov_half_angle", 0.6))
    # Sensing + filtering latency, in control steps.
    self._latency = int(p.get("latency_steps", 2))

    self._dt = env.step_dt
    n, dev = env.num_envs, env.device

    # World-frame XY estimate state.
    self._est_pos_w = torch.zeros(n, 2, device=dev)
    self._est_vel_w = torch.zeros(n, 2, device=dev)

    # Latency ring buffer of past world-frame estimates: (L+1, N, 4).
    self._buf = torch.zeros(self._latency + 1, n, 4, device=dev)
    self._buf_ptr = 0

    # Per-episode domain-randomized noise band.
    self._sigma_pos_scale = torch.ones(n, device=dev)
    self._sigma_vel = torch.full((n,), self._sigma_vel_range[0], device=dev)
    self._p_miss = torch.full((n,), self._p_miss_range[0], device=dev)

    self.reset(None)

  # -- geometry helpers -----------------------------------------------------

  def _gt_world(self) -> tuple[torch.Tensor, torch.Tensor]:
    ball = self._env.scene["ball"]
    return ball.data.root_link_pos_w[:, :2], ball.data.root_link_lin_vel_w[:, :2]

  def _to_body(self, pos_w: torch.Tensor, vel_w: torch.Tensor) -> torch.Tensor:
    """World-frame XY (pos, vel) -> body-frame (N, 4) using the current robot yaw."""
    robot = self._env.scene["robot"]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    yaw = _yaw_from_quat(robot.data.root_link_quat_w)
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    rel = pos_w - robot_pos
    px = c * rel[:, 0] - s * rel[:, 1]
    py = s * rel[:, 0] + c * rel[:, 1]
    vx = c * vel_w[:, 0] - s * vel_w[:, 1]
    vy = s * vel_w[:, 0] + c * vel_w[:, 1]
    return torch.stack([px, py, vx, vy], dim=-1)

  # -- lifecycle ------------------------------------------------------------

  def _sample_dr(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    dev = self._env.device

    def u(lo: float, hi: float) -> torch.Tensor:
      return torch.rand(n, device=dev) * (hi - lo) + lo

    self._sigma_pos_scale[env_ids] = u(*self._sigma_pos_scale_range)
    self._sigma_vel[env_ids] = u(*self._sigma_vel_range)
    self._p_miss[env_ids] = u(*self._p_miss_range)

  def reset(self, env_ids: torch.Tensor | None) -> dict:
    if env_ids is None:
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    pos_w, vel_w = self._gt_world()
    self._est_pos_w[env_ids] = pos_w[env_ids]
    self._est_vel_w[env_ids] = vel_w[env_ids]
    # A freshly reset env has no history: seed its whole latency buffer with GT.
    world_gt = torch.cat([pos_w, vel_w], dim=-1)
    self._buf[:, env_ids, :] = world_gt[env_ids].unsqueeze(0)
    self._sample_dr(env_ids)
    return {}

  # -- per-step update ------------------------------------------------------

  def _update(self) -> torch.Tensor:
    pos_w, vel_w = self._gt_world()
    robot = self._env.scene["robot"]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    n, dev = self._env.num_envs, self._env.device

    rel = pos_w - robot_pos
    rng = rel.norm(dim=-1)
    bearing = torch.atan2(rel[:, 1], rel[:, 0]) - _yaw_from_quat(
      robot.data.root_link_quat_w
    )
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

    in_fov = (bearing.abs() < self._fov_half) & (rng < self._max_range)
    too_close = rng < self._close_range
    rand_miss = torch.rand(n, device=dev) < self._p_miss
    seen = in_fov & ~too_close & ~rand_miss
    seen_f = seen.float().unsqueeze(-1)

    sigma_pos = (
      self._sigma_pos_base + self._sigma_pos_per_m * rng
    ) * self._sigma_pos_scale
    meas_pos = pos_w + torch.randn(n, 2, device=dev) * sigma_pos.unsqueeze(-1)
    meas_vel = vel_w + torch.randn(n, 2, device=dev) * self._sigma_vel.unsqueeze(-1)

    # Seen: snap position to the measurement, exp-filter velocity (lag).
    pos_seen = meas_pos
    vel_seen = (
      1.0 - self._vel_filter_alpha
    ) * self._est_vel_w + self._vel_filter_alpha * meas_vel
    # Unseen: constant-velocity coast with mild decay.
    pos_unseen = self._est_pos_w + self._est_vel_w * self._dt
    vel_unseen = self._est_vel_w * self._coast_vel_decay

    self._est_pos_w = seen_f * pos_seen + (1.0 - seen_f) * pos_unseen
    self._est_vel_w = seen_f * vel_seen + (1.0 - seen_f) * vel_unseen

    # Push current world estimate, read the L-step-delayed one back out.
    self._buf[self._buf_ptr] = torch.cat([self._est_pos_w, self._est_vel_w], dim=-1)
    self._buf_ptr = (self._buf_ptr + 1) % (self._latency + 1)
    delayed = self._buf[self._buf_ptr]
    return self._to_body(delayed[:, :2], delayed[:, 2:])

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    # This term lives in a single obs group, so it is called exactly once per
    # step: advancing state on every call keeps the latency/coast in lockstep
    # with the simulation, no step-counter guard needed.
    del env, kwargs
    return self._update()
