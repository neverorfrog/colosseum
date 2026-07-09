"""Stateful ball-state estimator that mimics the on-robot perception pipeline.

On the robot, the policy never sees the true ball state: it reads ``LocalBall``,
the output of ``WorldModelLocal`` (YOLO ``RawDetections`` -> per-track
``KalmanFilter3D`` -> ``LocalBall``), which is a filtered, lagged,
occasionally-stale estimate kept in the robot's local frame.

``BallPerceptionModel`` reproduces the *output* characteristics of that pipeline
on top of the simulator's GT ball position, matching the real tracker's design:

  * detections arrive at the camera rate (``detection_rate_hz``, 30 Hz on Maximus),
    not every control step -- between frames the estimate only predicts (coasts);
  * the *position* is measured with range-scaled Gaussian jitter (capped like
    ``measurementCovariance``'s ``clamp(..., 0.35)``) PLUS a per-episode systematic
    bias (calibration/extrinsics error the covariance never captures); velocity is
    never measured;
  * an alpha-beta filter -- the steady-state stand-in for the real per-track
    constant-velocity ``KalmanFilter3D`` -- smooths position and *infers* velocity
    from the position residual, so velocity is lagged and its noise is inherited
    from the position noise (not read from GT);
  * friction velocity decay (``1 - friction * dt``, ``friction = 0.6`` as in the
    ball's ``ObjectTypeParams``), applied every step, seen or not;
  * detection dropout: geometric visibility (out of FOV / too close under the body)
    plus a bursty two-state (Gilbert-Elliott) outage model, so the ball is lost for
    *stretches* of consecutive frames (motion blur / occlusion), not just isolated
    single-frame misses. The belief never disappears -- it coasts, like the
    persisted ``lastBall*`` in WorldModelLocal;
  * fixed sensing+filtering latency (ring buffer).

The state is kept in the *world* frame and transformed to the robot body frame
with the CURRENT robot pose at output time -- the analog of the real tracker's
odometry compensation -- so a stale (coasting) estimate correctly stays fixed in
the world while the robot turns.

The per-episode noise band (jitter scale, bias, outage frequency/length) is
domain-randomized in ``reset`` so the policy is robust across the band rather
than tuned to one operating point.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers import ObservationTermCfg

from colosseum.tasks.kick.mdp.observations import (
  ball_position,
  ball_velocity_xy,
)
from colosseum.tasks.kick.mdp.head_ik_action import (
  HEAD_FOV_HALF,
  HEAD_MAX_RANGE,
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
  """Clean GT body-frame ball state (N, 4) = [px, py, vx, vy].

  The privileged critic view: perfect position/velocity, so its layout matches
  the noisy actor estimate.
  """
  return torch.cat([ball_position(env), ball_velocity_xy(env)], dim=-1)


class BallPerceptionModel:
  """Noisy, lagged, droppable body-frame ball-state estimate. Returns (N, 4)."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    p = cfg.params

    # Position jitter: sigma = clamp((base + per_m*range) * per-episode scale, 0.35),
    # matching measurementCovariance's clamp(0.04+0.03*d, 0.35). Randomized per step.
    self._sigma_pos_base = float(p.get("sigma_pos_base", 0.04))
    self._sigma_pos_per_m = float(p.get("sigma_pos_per_m", 0.03))
    self._sigma_pos_scale_range = tuple(p.get("sigma_pos_scale_range", (0.5, 1.5)))
    self._sigma_pos_max = float(p.get("sigma_pos_max", 0.35))
    # Per-episode systematic position bias std (m): a constant offset on the whole
    # episode, on top of the per-step jitter -- error the covariance doesn't model.
    self._pos_bias_std = float(p.get("pos_bias_std", 0.05))
    # Alpha-beta filter gains (steady-state stand-in for the CV Kalman filter):
    # alpha corrects position, beta/dt corrects velocity from the position residual.
    self._alpha = float(p.get("pos_gain_alpha", 0.5))
    self._beta = float(p.get("vel_gain_beta", 0.15))
    # Velocity friction decay (1/s), applied every step like the ball's KF predict.
    self._friction = float(p.get("friction", 0.6))
    # Camera/detection rate (Hz): new detections only arrive this often; the filter
    # coasts on the control-rate steps in between.
    self._detection_rate_hz = float(p.get("detection_rate_hz", 30.0))
    # Bursty (Gilbert-Elliott) outage: per-episode mean burst length (s) and the
    # fraction of detection frames spent in an outage.
    self._outage_mean_s_range = tuple(p.get("outage_mean_s_range", (0.15, 0.6)))
    self._outage_frac_range = tuple(p.get("outage_frac_range", (0.1, 0.5)))
    # Isolated single-frame miss on top of the bursts (salt-and-pepper).
    self._p_miss_range = tuple(p.get("p_miss_range", (0.02, 0.1)))
    self._close_range = float(p.get("close_range_dropout", 0.3))
    # FOV matches the head's reach (the camera rides the tracking head), shared
    # with head IK and the ball-lost termination via the HEAD_* constants.
    self._max_range = float(p.get("max_range", HEAD_MAX_RANGE))
    self._fov_half = float(p.get("fov_half_angle", HEAD_FOV_HALF))
    # Sensing + filtering latency, in control steps.
    self._latency = int(p.get("latency_steps", 2))

    self._dt = env.step_dt
    self._decay = max(0.0, 1.0 - self._friction * self._dt)
    self._det_period = 1.0 / self._detection_rate_hz
    n, dev = env.num_envs, env.device

    # World-frame XY estimate state.
    self._est_pos_w = torch.zeros(n, 2, device=dev)
    self._est_vel_w = torch.zeros(n, 2, device=dev)

    # Detection cadence phase + bursty-outage state.
    self._det_accum = torch.zeros(n, device=dev)
    self._in_outage = torch.zeros(n, dtype=torch.bool, device=dev)

    # Latency ring buffer of past world-frame estimates: (L+1, N, 4).
    self._buf = torch.zeros(self._latency + 1, n, 4, device=dev)
    self._buf_ptr = 0

    # Per-episode domain-randomized band.
    self._sigma_pos_scale = torch.ones(n, device=dev)
    self._p_miss = torch.full((n,), self._p_miss_range[0], device=dev)
    self._p_enter = torch.zeros(n, device=dev)
    self._p_exit = torch.ones(n, device=dev)
    self._pos_bias = torch.zeros(n, 2, device=dev)

    self.reset(None)

  # -- geometry helpers -----------------------------------------------------

  def _gt_pos_world(self) -> torch.Tensor:
    return self._env.scene["ball"].data.root_link_pos_w[:, :2]

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
    self._p_miss[env_ids] = u(*self._p_miss_range)
    self._pos_bias[env_ids] = torch.randn(n, 2, device=dev) * self._pos_bias_std

    # Per-detection-frame outage transition probs from mean burst length + duty:
    # p_exit = 1 / mean_frames, p_enter = p_exit * frac / (1 - frac).
    mean_s = u(*self._outage_mean_s_range)
    frac = u(*self._outage_frac_range)
    p_exit = (self._det_period / mean_s).clamp(max=1.0)
    p_enter = (p_exit * frac / (1.0 - frac)).clamp(max=1.0)
    self._p_exit[env_ids] = p_exit
    self._p_enter[env_ids] = p_enter

  def reset(self, env_ids: torch.Tensor | None) -> dict:
    if env_ids is None:
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    dev = self._env.device
    pos_w = self._gt_pos_world()
    self._est_pos_w[env_ids] = pos_w[env_ids]
    # Velocity starts at zero and ramps up via the filter, like KalmanFilter3D::init.
    self._est_vel_w[env_ids] = 0.0
    self._in_outage[env_ids] = False
    # Stagger the detection phase so envs don't all refresh on the same step.
    self._det_accum[env_ids] = torch.rand(len(env_ids), device=dev) * self._det_period
    # A freshly reset env has no history: seed its whole latency buffer with the
    # zero-velocity state.
    seed = torch.zeros(len(env_ids), 4, device=dev)
    seed[:, :2] = pos_w[env_ids]
    self._buf[:, env_ids, :] = seed.unsqueeze(0)
    self._sample_dr(env_ids)
    return {}

  # -- per-step update ------------------------------------------------------

  def _update(self) -> torch.Tensor:
    pos_w = self._gt_pos_world()
    robot = self._env.scene["robot"]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    n, dev = self._env.num_envs, self._env.device

    rel = pos_w - robot_pos
    rng = rel.norm(dim=-1)
    bearing = torch.atan2(rel[:, 1], rel[:, 0]) - _yaw_from_quat(
      robot.data.root_link_quat_w
    )
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

    # Detection cadence: a new camera frame only every ``_det_period`` seconds.
    self._det_accum += self._dt
    tick = self._det_accum >= self._det_period
    self._det_accum = torch.where(
      tick, self._det_accum - self._det_period, self._det_accum
    )

    # Bursty outage: advance the two-state chain only on detection frames.
    enter = torch.rand(n, device=dev) < self._p_enter
    stay = ~(torch.rand(n, device=dev) < self._p_exit)
    next_outage = torch.where(self._in_outage, stay, enter)
    self._in_outage = torch.where(tick, next_outage, self._in_outage)

    in_fov = (bearing.abs() < self._fov_half) & (rng < self._max_range)
    too_close = rng < self._close_range
    rand_miss = torch.rand(n, device=dev) < self._p_miss
    seen = tick & in_fov & ~too_close & ~self._in_outage & ~rand_miss
    seen_f = seen.float().unsqueeze(-1)

    # Position measurement: per-step jitter (capped) + per-episode systematic bias.
    sigma_pos = (
      (self._sigma_pos_base + self._sigma_pos_per_m * rng) * self._sigma_pos_scale
    ).clamp(max=self._sigma_pos_max)
    meas_pos = (
      pos_w + self._pos_bias + torch.randn(n, 2, device=dev) * sigma_pos.unsqueeze(-1)
    )

    # Alpha-beta predict (constant velocity + friction decay).
    pos_pred = self._est_pos_w + self._est_vel_w * self._dt
    vel_pred = self._est_vel_w * self._decay

    # Correct from the position residual when seen; else coast on the prediction.
    resid = meas_pos - pos_pred
    pos_corr = pos_pred + self._alpha * resid
    vel_corr = vel_pred + (self._beta / self._dt) * resid

    self._est_pos_w = seen_f * pos_corr + (1.0 - seen_f) * pos_pred
    self._est_vel_w = seen_f * vel_corr + (1.0 - seen_f) * vel_pred

    # Push current world estimate, read the L-step-delayed one back out.
    self._buf[self._buf_ptr] = torch.cat([self._est_pos_w, self._est_vel_w], dim=-1)
    self._buf_ptr = (self._buf_ptr + 1) % (self._latency + 1)
    delayed = self._buf[self._buf_ptr]
    return self._to_body(delayed[:, :2], delayed[:, 2:4])

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    # This term lives in a single obs group, so it is called exactly once per
    # step: advancing state on every call keeps the latency/coast in lockstep
    # with the simulation, no step-counter guard needed.
    del env, kwargs
    return self._update()
