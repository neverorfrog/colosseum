"""Ball dribbling/kicking reward functions.

Primary task reference: Ji et al., "DribbleBot" (ICRA 2023).
Ball velocity rewards split into three terms (TABLE III):
  - ball_vel_tracking: full vector error exp(-δ|v^b - v^cmd|²)
  - ball_vel_norm:     speed matching  exp(-δ(|v^cmd| - |v^b|)²)
  - ball_vel_angle:    direction match 1 - (ψ_b - ψ_cmd)²/π²

Body-frame variants (``_body`` suffix) rotate both ball velocity and command
into the robot body frame before comparison.  The tracking and angular rewards
are rotation-invariant — numerically identical to world-frame — but computing
them in body frame keeps the reward signal consistent with the actor's
body-frame observations (command, ball position, ball velocity from encoder).
``ball_vel_norm`` has no body-frame variant because it only compares magnitudes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from colosseum.tasks.dribbling.mdp.obstacle_commands import ObstacleCommand

# ------------------------------------------------------------------
# Body-frame helpers (shared by tracking, angle, and yaw rewards)
# ------------------------------------------------------------------


def _ball_vel_body(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball XY velocity in robot body frame. Shape (N, 2)."""
  ball_vel_w = env.scene["ball"].data.root_link_lin_vel_w[:, :3]
  quat_w = env.scene["robot"].data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  return quat_apply(quat_conj, ball_vel_w)[:, :2]


def _cmd_body(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Ball velocity command rotated into robot body frame. Shape (N, 2)."""
  cmd_w = env.command_manager.get_command(command_name)[:, :2]
  quat_w = env.scene["robot"].data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  cmd_3d = torch.cat([cmd_w, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)
  return quat_apply(quat_conj, cmd_3d)[:, :2]


# ------------------------------------------------------------------
# Primary task rewards (DribbleBot TABLE III) — body-frame variants
# ------------------------------------------------------------------


def ball_vel_tracking_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
) -> torch.Tensor:
  """exp(-sharpness * |v_ball_b - v_cmd_b|²). Body-frame variant."""
  error_sq = ((_ball_vel_body(env) - _cmd_body(env, command_name)) ** 2).sum(dim=-1)
  return torch.exp(-sharpness * error_sq)


def ball_vel_angle_body(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Direction match 1 - (ψ_ball - ψ_cmd)²/π². Body-frame variant."""
  ball_vel_b = _ball_vel_body(env)
  cmd_b = _cmd_body(env, command_name)
  psi_ball = torch.atan2(ball_vel_b[:, 1], ball_vel_b[:, 0])
  psi_cmd = torch.atan2(cmd_b[:, 1], cmd_b[:, 0])
  angle_err = (psi_ball - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  return 1.0 - (angle_err**2) / (math.pi**2)


def robot_ball_yaw_body(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Ball ahead of robot along command direction AND robot facing that way (body frame).

  e1 = 1 - dot(d_robot→ball_b, cmd_dir_b)  (ball in command direction)
  e2 = 1 - d_robot→ball_b[0] / |d|         (ball in front, X-forward)
  reward = exp(-2 * (e1 + e2))
  """
  robot = env.scene["robot"]
  ball_pos_w = env.scene["ball"].data.root_link_pos_w[:, :3]
  robot_pos_w = robot.data.root_link_pos_w[:, :3]

  relative_w = ball_pos_w - robot_pos_w
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  d_ball_b = quat_apply(quat_conj, relative_w)[:, :2]
  d_norm = d_ball_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  d_ball_b_unit = d_ball_b / d_norm

  cmd_b = _cmd_body(env, command_name)
  unit_cmd = cmd_b / cmd_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  e1 = 1.0 - (d_ball_b_unit * unit_cmd).sum(dim=-1)
  e2 = 1.0 - d_ball_b_unit[:, 0]  # dot with [1, 0] (X-forward in body frame)

  reward = torch.exp(-2.0 * (e1 + e2))
  return reward * (cmd_b.norm(dim=-1) > min_speed).float()


# ------------------------------------------------------------------
# Primary task rewards (DribbleBot TABLE III) — world-frame originals
# (kept for reference; use _body variants in configs)
# ------------------------------------------------------------------


def ball_vel_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
) -> torch.Tensor:
  """Full XY velocity vector tracking: exp(-sharpness * |v^b - v^cmd|²).

  Low weight (0.5) — hardest to achieve, but penalises both speed and
  direction error simultaneously.
  """
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_vel = env.command_manager.get_command(command_name)[:, :2]  # type: ignore
  error_sq = ((ball_vel - target_vel) ** 2).sum(dim=-1)
  return torch.exp(-sharpness * error_sq)


def ball_vel_norm(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
) -> torch.Tensor:
  """Speed matching: exp(-sharpness * (|v^cmd| - |v^b|)²).

  Rewards matching commanded speed regardless of direction.
  Prevents exploitation of direction-only rewards by kicking too hard.
  """
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_vel = env.command_manager.get_command(command_name)[:, :2]  # type: ignore
  speed_err = (target_vel.norm(dim=-1) - ball_vel.norm(dim=-1)) ** 2
  return torch.exp(-sharpness * speed_err)


def ball_vel_angle(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Direction match: 1 - (ψ_b - ψ_cmd)²/π².

  Ranges from 1.0 (perfect alignment) to 0.0 (opposite direction).
  Gives partial credit for near-correct directions.
  """
  ball_vel = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_vel = env.command_manager.get_command(command_name)[:, :2]  # type: ignore
  psi_b = torch.atan2(ball_vel[:, 1], ball_vel[:, 0])
  psi_cmd = torch.atan2(target_vel[:, 1], target_vel[:, 0])
  angle_err = (psi_b - psi_cmd + math.pi) % (2 * math.pi) - math.pi
  return 1.0 - (angle_err**2) / (math.pi**2)


# ------------------------------------------------------------------
# Phase-schedule feet rewards (DribbleBot TABLE III)
# ------------------------------------------------------------------


def swing_phase_schedule(
  env: ManagerBasedRlEnv,
  phase_command_name: str,
  sensor_name: str,
  sharpness: float = 0.1,
) -> torch.Tensor:
  """During swing phase, penalize foot-ground contact force.

  reward = sum_feet( [1 - κ] * exp(-sharpness * |f_foot|²) )
  κ = (1 + cos(φ)) / 2  →  0 in full swing, 1 in full stance.
  """
  phase = env.command_manager.get_command(
    phase_command_name
  )  # (N, 4): [cL, cR, sL, sR]
  kappa = (1.0 + phase[:, :2]) / 2.0  # (N, 2)
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  force_sq = (contact_sensor.data.force**2).sum(dim=-1)  # (N, 2)
  return ((1.0 - kappa) * torch.exp(-sharpness * force_sq)).sum(dim=-1)


def stance_phase_schedule(
  env: ManagerBasedRlEnv,
  phase_command_name: str,
  asset_cfg: SceneEntityCfg,
  sharpness: float = 0.1,
) -> torch.Tensor:
  """During stance phase, penalize foot XY sliding velocity.

  reward = sum_feet( κ * exp(-sharpness * |v_foot_xy|²) )
  """
  phase = env.command_manager.get_command(phase_command_name)  # (N, 4)
  kappa = (1.0 + phase[:, :2]) / 2.0  # (N, 2)
  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
  vel_sq = (foot_vel_xy**2).sum(dim=-1)  # (N, 2)
  return (kappa * torch.exp(-sharpness * vel_sq)).sum(dim=-1)


# ------------------------------------------------------------------
# Pose deviation
# ------------------------------------------------------------------


def pose_deviation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  std: float,
) -> torch.Tensor:
  """Penalize joint deviation from default pose: exp(-mean(error²/std²)).

  Matches variable_posture's formula with a single scalar std.
  Register separate terms for arms and legs with different std and weight.
  Smaller std = tighter constraint.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.exp(-torch.mean(torch.square(q - q_default) / (std**2), dim=1))


# ------------------------------------------------------------------
# Feet distance penalty
# ------------------------------------------------------------------


def feet_distance_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_dist: float = 0.2,
) -> torch.Tensor:
  """Penalize when feet are closer than min_dist (XY plane).

  penalty = clip(min_dist - ||p_left_xy - p_right_xy||, 0, min_dist)

  Gives a continuous repulsive gradient before feet actually collide.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :3]  # (N, 2, 3)
  base_pos_w = asset.data.root_link_pos_w[:, :3].unsqueeze(1)  # (N, 1, 3)
  quat_w = asset.data.root_link_quat_w  # (N, 4)
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)

  # Transform each foot into body frame
  left_b = quat_apply(quat_conj, foot_pos_w[:, 0] - base_pos_w[:, 0])  # (N, 3)
  right_b = quat_apply(quat_conj, foot_pos_w[:, 1] - base_pos_w[:, 0])  # (N, 3)
  dist = (left_b[:, 1] - right_b[:, 1]).abs()  # (N,) — Y axis only
  return (min_dist - dist).clamp(min=0.0, max=min_dist)


# ------------------------------------------------------------------
# Robot–ball spatial relationship
# ------------------------------------------------------------------


def robot_ball_distance(
  env: ManagerBasedRlEnv,
  sharpness: float = 2.0,
) -> torch.Tensor:
  """exp(-sharpness * ||robot_xy - ball_xy||²).

  Low sharpness (0.5) → half-max at ~1.2m so the robot can play the ball
  forward into free areas without being penalized.  High sharpness (2.0)
  keeps the ball at feet (~0.6m half-max), suitable for tight dribbling.
  """
  robot_pos = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  dist_sq = ((ball_pos - robot_pos) ** 2).sum(dim=-1)
  return torch.exp(-sharpness * dist_sq)


def robot_ball_yaw(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_speed: float = 0.05,
) -> torch.Tensor:
  """Ball ahead of robot along command direction AND robot body faces that direction.

  e1 = 1 - dot(d_robot→ball, d̂_cmd)   (ball not ahead)
  e2 = 1 - dot(d_robot→ball, body_fwd) (robot not facing cmd)
  reward = exp(-2 * (e1 + e2))
  """
  robot = env.scene["robot"]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  robot_pos = robot.data.root_link_pos_w[:, :2]

  target_vel = env.command_manager.get_command(command_name)[:, :2]  # type: ignore
  unit_cmd = target_vel / target_vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  d_robot_ball = ball_pos - robot_pos
  d_robot_ball = d_robot_ball / d_robot_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  e1 = 1.0 - (d_robot_ball * unit_cmd).sum(dim=-1)

  q = robot.data.root_link_quat_w
  yaw = torch.atan2(
    2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
  )
  body_fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
  e2 = 1.0 - (d_robot_ball * body_fwd).sum(dim=-1)

  reward = torch.exp(-2.0 * (e1 + e2))
  return reward * (target_vel.norm(dim=-1) > min_speed).float()


def _project_ball_to_camera(
  env: ManagerBasedRlEnv,
  camera_name: str,
  aspect_ratio: float,
  head_camera_fovy: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Project ball world position into normalised camera image coordinates.

  Returns:
    nx:       [num_envs] horizontal coord, -1 = left edge, +1 = right edge
    ny:       [num_envs] vertical coord,   -1 = bottom edge, +1 = top edge
    in_front: [num_envs] bool mask — True when ball is in front of camera

  MuJoCo camera frame convention: x = right, y = up, z = backward.
  The optical axis is -z, so depth = -p_cam.z (positive in front).
  cam_xmat rows are the camera's local axes expressed in world frame,
  so   p_cam = cam_mat @ (p_ball - cam_pos)   transforms world → camera.
  """

  cam_id = env.sim.mj_model.camera(camera_name).id
  cam_pos = env.sim.data.cam_xpos[:, cam_id, :]  # [N, 3]
  # cam_xmat columns are camera-frame axes in world coordinates (R_world_from_cam).
  # To transform world→camera we need R^T: p_cam = R^T @ p_rel.
  cam_mat = env.sim.data.cam_xmat[:, cam_id, :].reshape(-1, 3, 3)  # [N, 3, 3]

  ball_pos = env.scene["ball"].data.root_link_pos_w  # [N, 3]
  p_rel = ball_pos - cam_pos  # [N, 3]

  # Rotate into camera frame: p_cam = R^T @ p_rel
  p_cam = torch.bmm(cam_mat.transpose(1, 2), p_rel.unsqueeze(-1)).squeeze(-1)  # [N, 3]

  # Depth along optical axis (positive means ball is in front)
  depth = (-p_cam[:, 2]).clamp_min(1e-6)  # [N]
  in_front = p_cam[:, 2] < 0  # [N] bool

  # Perspective-divide to get tangent of the off-axis angles
  tx = p_cam[:, 0] / depth  # tan(horizontal angle), right = positive
  ty = p_cam[:, 1] / depth  # tan(vertical angle),   up    = positive

  # Normalize by FOV half-extents so ±1 = edge of frame
  tan_half_v = math.tan(math.radians(head_camera_fovy / 2))
  tan_half_h = tan_half_v * aspect_ratio

  nx = tx / tan_half_h
  ny = ty / tan_half_v

  return nx, ny, in_front


def ball_projection(
  env: ManagerBasedRlEnv,
  camera_name: str = "robot/d455_color",
  aspect_ratio: float = 4.0 / 3.0,
) -> torch.Tensor:
  """Normalised ball position in camera image frame — observation term.

  Returns a [num_envs, 2] tensor of (nx, ny) ∈ [-1, 1]² that can be added
  directly to the policy observation.  At deployment these values are
  obtained from YOLO bounding-box centre coordinates via:
      nx = (u / W - 0.5) / 0.5,   ny = (0.5 - v / H) / 0.5

  When the ball is behind the camera both values are clamped to ±2 so the
  policy can distinguish "ball behind me" from "ball at the edge of frame".
  """
  nx, ny, in_front = _project_ball_to_camera(env, camera_name, aspect_ratio)
  # Out-of-front values are clamped rather than zeroed so the policy keeps
  # a gradient signal even when the ball is barely behind the camera plane.
  nx = torch.where(in_front, nx, nx.clamp(-2.0, 2.0))
  ny = torch.where(in_front, ny, ny.clamp(-2.0, 2.0))
  return torch.stack([nx, ny], dim=-1)  # [num_envs, 2]


def head_ball_tracking(
  env: ManagerBasedRlEnv,
  camera_name: str = "robot/d455_color",
) -> torch.Tensor:
  """Cosine similarity between camera optical axis and camera→ball direction.

  reward = (1 + dot(cam_fwd, unit_to_ball)) / 2
    1.0 when perfectly aligned, 0.5 when perpendicular, 0.0 when opposite.

  Unlike ball_in_camera_fov (which zeros when ball is behind), this term has
  nonzero gradient everywhere — so the policy always has a signal to rotate
  the head toward the ball even when ball is fully outside the FOV.
  """
  cam_id = env.sim.mj_model.camera(camera_name).id
  cam_pos = env.sim.data.cam_xpos[:, cam_id, :]  # [N, 3]
  cam_mat = env.sim.data.cam_xmat[:, cam_id, :].reshape(-1, 3, 3)  # [N, 3, 3]

  ball_pos = env.scene["ball"].data.root_link_pos_w  # [N, 3]
  to_ball = ball_pos - cam_pos
  to_ball = to_ball / to_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  # cam_xmat columns are camera axes in world frame (col 2 = z = backward).
  # Optical axis forward = -z_column.
  cam_fwd = -cam_mat[:, :, 2]  # [N, 3]
  cos_sim = (cam_fwd * to_ball).sum(dim=-1).clamp(-1.0, 1.0)
  return (1.0 + cos_sim) / 2.0


def robot_ball_approach_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Robot base moving toward ball fast enough (penalizes only speed deficit).

  deficit = max(0, cmd_speed - proj(robot_vel, d_robot→ball))
  reward  = exp(-deficit²)
  """
  robot = env.scene["robot"]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  robot_pos = robot.data.root_link_pos_w[:, :2]

  d_robot_ball = ball_pos - robot_pos
  d_robot_ball = d_robot_ball / d_robot_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  approach_vel = (robot.data.root_link_lin_vel_w[:, :2] * d_robot_ball).sum(dim=-1)
  cmd_speed = env.command_manager.get_command(command_name)[:, :2].norm(dim=-1)  # type: ignore
  deficit = (cmd_speed - approach_vel).clamp(min=0.0)
  return torch.exp(-(deficit**2))


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------


def _get_min_robot_obstacle_dist(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return (min_dist, nearest_obs_xy) for each env.

  min_dist  : (N,)    minimum XY distance from robot to any active obstacle.
  nearest   : (N, 2)  world-frame XY position of the nearest obstacle.
  """
  robot = env.scene["robot"]
  term: ObstacleCommand = env.command_manager.get_term(command_name)
  robot_xy = robot.data.root_link_pos_w[:, :2]  # (N, 2)
  obs_xy = term.obstacle_positions_w  # (N, K, 2)

  dist = (obs_xy - robot_xy.unsqueeze(1)).norm(dim=-1)  # (N, K)
  min_dist, idx = dist.min(dim=-1)  # (N,)
  nearest = obs_xy[torch.arange(env.num_envs, device=env.device), idx]  # (N, 2)
  return min_dist, nearest


def _get_closest_robot_obstacle(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return distance, position, and velocity of the nearest active obstacle."""
  robot = env.scene["robot"]
  term: ObstacleCommand = env.command_manager.get_term(command_name)
  robot_xy = robot.data.root_link_pos_w[:, :2]  # (N, 2)
  obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]  # (N, Ka, 2)
  obs_vel = term.obstacle_velocities_w[:, : term.cfg.num_active]  # (N, Ka, 2)

  dist = (obs_xy - robot_xy.unsqueeze(1)).norm(dim=-1)  # (N, Ka)
  min_dist, idx = dist.min(dim=-1)
  batch_idx = torch.arange(env.num_envs, device=env.device)
  nearest_xy = obs_xy[batch_idx, idx]
  nearest_vel = obs_vel[batch_idx, idx]
  return min_dist, nearest_xy, nearest_vel


def _world_xy_to_body_xy(
  env: ManagerBasedRlEnv,
  world_xy: torch.Tensor,
) -> torch.Tensor:
  """Rotate world-frame XY points into the robot body frame."""
  robot = env.scene["robot"]
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  robot_xy = robot.data.root_link_pos_w[:, :2]
  zeros = torch.zeros(env.num_envs, 1, device=env.device)
  rel_3d = torch.cat([world_xy - robot_xy, zeros], dim=-1)
  return quat_apply(quat_conj, rel_3d)[:, :2]


def obstacle_avoidance(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  ball_vel_command_name: str = "ball_vel",
  detection_range: float = 3.0,
  collision_sharpness: float = 2.0,
  direction_sharpness: float = 4.0,
  collision_weight: float = 0.4,
  direction_weight: float = 1.0,
  min_cmd_speed: float = 0.05,
  cmd_speed_ref: float = 1.0,
  ball_engagement_radius: float = 1.0,
) -> torch.Tensor:
  """Visible-obstacle penalty with collision and kick-direction terms.

  The nearest active obstacle contributes only when it is:
    - within ``detection_range`` from the robot, and
    - in front of the robot in body frame (local x > 0).

  The penalty is the sum of:
    1. collision term:   exp(-collision_sharpness * ||robot - obstacle||^2)
    2. direction term:   exp(direction_sharpness * max(0, cos(cmd, ball->obs))) - 1

  The first discourages direct collision with the obstacle. The second
  discourages commanding ball motion toward the obstacle, is weighted more
  heavily by default, and scales with the requested ball speed so fast
  commands into the obstacle are penalized more strongly than slow ones.
  The full obstacle penalty is also down-weighted when the robot is far from
  the ball, so obstacle shaping mainly acts during actual dribbling
  interactions instead of rewarding the robot for abandoning the ball.
  """
  term: ObstacleCommand = env.command_manager.get_term(command_name)
  if term.cfg.num_active == 0:
    return torch.zeros(env.num_envs, device=env.device)

  min_dist, nearest_obs_xy, _ = _get_closest_robot_obstacle(env, command_name)
  nearest_obs_b = _world_xy_to_body_xy(env, nearest_obs_xy)

  visible = (min_dist < detection_range) & (nearest_obs_b[:, 0] > 0.0)

  collision_term = torch.exp(-collision_sharpness * min_dist.pow(2))

  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  cmd_xy = env.command_manager.get_command(ball_vel_command_name)[:, :2]
  cmd_speed = cmd_xy.norm(dim=-1)
  cmd_dir = cmd_xy / cmd_speed.unsqueeze(-1).clamp(min=1e-6)

  ball_to_obs = nearest_obs_xy - ball_xy
  ball_to_obs_dir = ball_to_obs / ball_to_obs.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  alignment = (cmd_dir * ball_to_obs_dir).sum(dim=-1).clamp(-1.0, 1.0)
  toward_obstacle = alignment.clamp(min=0.0)
  direction_term = torch.exp(direction_sharpness * toward_obstacle) - 1.0
  direction_term = direction_term / (math.exp(direction_sharpness) - 1.0)
  direction_term = direction_term * (cmd_speed > min_cmd_speed).float()
  cmd_speed_scale = (cmd_speed / max(cmd_speed_ref, 1e-6)).clamp(min=0.0, max=1.0)
  direction_term = direction_term * cmd_speed_scale
  ball_dist = (ball_xy - robot_xy).norm(dim=-1)
  ball_engagement = torch.exp(-((ball_dist / max(ball_engagement_radius, 1e-6)) ** 2))

  penalty = collision_weight * collision_term + direction_weight * direction_term
  return visible.float() * ball_engagement * penalty


def ball_protection(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  activation_radius: float = 3.0,
) -> torch.Tensor:
  """Reward the robot for shielding the ball from the nearest obstacle.

  When the nearest obstacle is within *activation_radius*:
    - Computes the unit vector from the ball toward the nearest obstacle.
    - Computes the unit vector from the ball toward the robot.
    - Reward = activation_weight * (1 + cosine_similarity) / 2

  A score of 1.0 means the robot is perfectly interposed between the ball
  and the obstacle; 0.0 means it is on the opposite side.

  Returns shape (N,), values in [0, 1].
  """
  term: ObstacleCommand = env.command_manager.get_term(command_name)
  if term.cfg.num_active == 0:
    return torch.zeros(env.num_envs, device=env.device)

  robot = env.scene["robot"]
  ball = env.scene["ball"]

  robot_xy = robot.data.root_link_pos_w[:, :2]  # (N, 2)
  ball_xy = ball.data.root_link_pos_w[:, :2]  # (N, 2)

  _, nearest_obs_xy = _get_min_robot_obstacle_dist(env, command_name)
  dist_to_nearest = (nearest_obs_xy - robot_xy).norm(dim=-1)  # (N,)

  # Smooth activation: 1 at r=0, ~0.37 at r=activation_radius.
  sigma = activation_radius / 3.0
  activation = torch.exp(-(dist_to_nearest / sigma).pow(2))

  # Direction from ball toward obstacle.
  ball_to_obs = nearest_obs_xy - ball_xy  # (N, 2)
  ball_to_obs_norm = ball_to_obs / ball_to_obs.norm(dim=-1, keepdim=True).clamp(min=0.1)

  # Direction from ball toward robot.
  ball_to_robot = robot_xy - ball_xy  # (N, 2)
  ball_to_robot_norm = ball_to_robot / ball_to_robot.norm(dim=-1, keepdim=True).clamp(
    min=0.1
  )

  # Cosine similarity: +1 when robot is between ball and obstacle, -1 opposite.
  cos = (ball_to_robot_norm * ball_to_obs_norm).sum(dim=-1).clamp(-1.0, 1.0)

  # Also weight by how close the robot is to the ball (want robot near ball).
  ball_dist = ball_to_robot.norm(dim=-1).clamp(max=2.0)
  closeness = torch.exp(-ball_dist / 0.5)

  return activation * (1.0 + cos) / 2.0 * closeness


# ---------------------------------------------------------------------------
# Context-aware gating
# ---------------------------------------------------------------------------


def _compute_gates(
  env: ManagerBasedRlEnv,
  adversary_command_name: str = "adversary",
  ball_vel_command_name: str = "ball_vel",
  ball_far_threshold: float = 1.0,
  lookahead: float = 3.5,
  r_base: float = 0.5,
  k_speed: float = 0.5,
  gate_sharpness: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Smooth per-env gate scalars in [0, 1].

  danger : tube-intersection score for the nearest active obstacle.
    Tube runs along the commanded ball-velocity direction from the ball;
    radius = r_base + k_speed * obstacle_speed; capped at lookahead distance.
  ball_far : sigmoid gate that rises as robot–ball XY distance exceeds
    ball_far_threshold.
  """
  device = env.device
  N = env.num_envs

  robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  dist = (ball_xy - robot_xy).norm(dim=-1)
  ball_far = torch.sigmoid(gate_sharpness * (dist - ball_far_threshold))

  term: ObstacleCommand = env.command_manager.get_term(adversary_command_name)
  if term.cfg.num_active == 0:
    return torch.zeros(N, device=device), ball_far

  ball_vel_cmd = env.command_manager.get_command(ball_vel_command_name)[:, :2]
  cmd_dir = ball_vel_cmd / ball_vel_cmd.norm(dim=-1, keepdim=True).clamp(min=1e-3)

  _, nearest_obs_xy, nearest_obs_vel = _get_closest_robot_obstacle(
    env, adversary_command_name
  )
  obs_speed = nearest_obs_vel.norm(dim=-1)  # (N,)

  ball_to_obs = nearest_obs_xy - ball_xy  # (N, 2)
  proj = (ball_to_obs * cmd_dir).sum(dim=-1)  # (N,)
  d_perp = (ball_to_obs - proj.unsqueeze(-1) * cmd_dir).norm(dim=-1)  # (N,)

  r = r_base + k_speed * obs_speed  # (N,)
  inside_tube = torch.sigmoid(gate_sharpness * (r - d_perp))
  in_front = torch.sigmoid(gate_sharpness * proj)
  in_range = torch.sigmoid(gate_sharpness * (lookahead - proj))
  danger = inside_tube * in_front * in_range

  return danger, ball_far


def obstacle_avoidance_gated(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  ball_vel_command_name: str = "ball_vel",
  ball_far_threshold: float = 1.0,
  lookahead: float = 3.5,
  r_base: float = 0.5,
  k_speed: float = 0.5,
  gate_sharpness: float = 5.0,
  detection_range: float = 3.0,
  collision_sharpness: float = 2.0,
  direction_sharpness: float = 4.0,
  collision_weight: float = 0.4,
  direction_weight: float = 1.0,
  min_cmd_speed: float = 0.05,
  cmd_speed_ref: float = 1.0,
  ball_engagement_radius: float = 1.0,
) -> torch.Tensor:
  """obstacle_avoidance weighted by the danger tube gate."""
  danger, _ = _compute_gates(
    env, command_name, ball_vel_command_name,
    ball_far_threshold, lookahead, r_base, k_speed, gate_sharpness,
  )
  return danger * obstacle_avoidance(
    env,
    command_name=command_name,
    ball_vel_command_name=ball_vel_command_name,
    detection_range=detection_range,
    collision_sharpness=collision_sharpness,
    direction_sharpness=direction_sharpness,
    collision_weight=collision_weight,
    direction_weight=direction_weight,
    min_cmd_speed=min_cmd_speed,
    cmd_speed_ref=cmd_speed_ref,
    ball_engagement_radius=ball_engagement_radius,
  )


def ball_protection_gated(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  ball_vel_command_name: str = "ball_vel",
  activation_radius: float = 3.0,
  ball_far_threshold: float = 1.0,
  lookahead: float = 3.5,
  r_base: float = 0.5,
  k_speed: float = 0.5,
  gate_sharpness: float = 5.0,
) -> torch.Tensor:
  """ball_protection weighted by the danger tube gate."""
  danger, _ = _compute_gates(
    env, command_name, ball_vel_command_name,
    ball_far_threshold, lookahead, r_base, k_speed, gate_sharpness,
  )
  return danger * ball_protection(env, command_name, activation_radius)


def robot_ball_distance_gated(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  ball_vel_command_name: str = "ball_vel",
  sharpness_base: float = 0.5,
  sharpness_tight: float = 2.0,
  ball_far_loosening: float = 0.7,
  ball_far_threshold: float = 1.0,
  lookahead: float = 3.5,
  r_base: float = 0.5,
  k_speed: float = 0.5,
  gate_sharpness: float = 5.0,
) -> torch.Tensor:
  """robot_ball_distance with sharpness tight under danger, loose when ball_far.

  effective_sharpness = (base + (tight - base) * danger) * (1 - loosening * ball_far)
  """
  danger, ball_far = _compute_gates(
    env, command_name, ball_vel_command_name, ball_far_threshold,
    lookahead, r_base, k_speed, gate_sharpness,
  )
  effective_sharpness = (
    sharpness_base + (sharpness_tight - sharpness_base) * danger
  ) * (1.0 - ball_far_loosening * ball_far)
  effective_sharpness = effective_sharpness.clamp(min=0.05)
  robot_pos = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_pos = env.scene["ball"].data.root_link_pos_w[:, :2]
  dist_sq = ((ball_pos - robot_pos) ** 2).sum(dim=-1)
  return torch.exp(-effective_sharpness * dist_sq)


def robot_ball_approach_vel_gated(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  ball_vel_command_name: str = "ball_vel",
  ball_far_threshold: float = 1.0,
  lookahead: float = 3.5,
  r_base: float = 0.5,
  k_speed: float = 0.5,
  gate_sharpness: float = 5.0,
) -> torch.Tensor:
  """robot_ball_approach_vel gated by ball_far * (1 - danger)."""
  danger, ball_far = _compute_gates(
    env, command_name, ball_vel_command_name, ball_far_threshold,
    lookahead, r_base, k_speed, gate_sharpness,
  )
  return ball_far * (1.0 - danger) * robot_ball_approach_vel(env, ball_vel_command_name)
