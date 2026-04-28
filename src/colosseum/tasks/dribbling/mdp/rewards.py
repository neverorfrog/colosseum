"""Dribbling-specific reward functions.

Generic ball/locomotion rewards live in colosseum/mdp/ball_rewards.py and
colosseum/mdp/rewards.py and are re-exported here for backward compatibility.
This module keeps only the obstacle-aware and dribbling-specific terms.

Primary task reference: Ji et al., "DribbleBot" (ICRA 2023).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from colosseum.tasks.dribbling.mdp.obstacle_commands import ObstacleCommand

# Re-export generic ball rewards (avoid cross-task imports into dribbling)
from colosseum.mdp.ball_rewards import (  # noqa: F401
  ball_vel_tracking,
  ball_vel_tracking_body,
  ball_vel_norm,
  ball_vel_angle,
  ball_vel_angle_body,
  robot_ball_yaw_body,
  robot_ball_distance,
  robot_ball_approach_vel,
  camera_fov_mask as _camera_fov_mask_impl,
  ball_vel_body as _ball_vel_body,
  cmd_body as _cmd_body,
)

# Re-export locomotion rewards
from colosseum.mdp.rewards import (  # noqa: F401
  pose_deviation,
  feet_distance_penalty,
  swing_phase_schedule,
  stance_phase_schedule,
)

# Keep world-frame yaw reward under old name for backward compat
robot_ball_yaw = robot_ball_yaw_body  # noqa: F811  (world-frame alias, body-frame is canonical)


def _camera_fov_mask(
  env: ManagerBasedRlEnv,
  pos_w: torch.Tensor,
  camera_name: str,
  camera_fovy: float,
  camera_aspect_ratio: float,
  depth_clip: float,
) -> torch.Tensor:
  """Thin wrapper around colosseum.mdp.ball_rewards.camera_fov_mask."""
  return _camera_fov_mask_impl(env, pos_w, camera_name, camera_fovy, camera_aspect_ratio, depth_clip)


# ---------------------------------------------------------------------------
# Obstacle helpers
# ---------------------------------------------------------------------------


def _get_closest_robot_obstacle(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return distance, position, and velocity of the nearest active obstacle."""
  robot = env.scene["robot"]
  term: ObstacleCommand = env.command_manager.get_term(command_name)
  robot_xy = robot.data.root_link_pos_w[:, :2]
  obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]
  obs_vel = term.obstacle_velocities_w[:, : term.cfg.num_active]

  dist = (obs_xy - robot_xy.unsqueeze(1)).norm(dim=-1)
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


def _ball_engagement_gate(
  env: ManagerBasedRlEnv,
  ball_engagement_near_distance: float,
  ball_engagement_far_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return robot-ball distance and a [0, 1] engagement gate."""
  robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  ball_dist = (ball_xy - robot_xy).norm(dim=-1)
  engagement_span = max(
    ball_engagement_far_distance - ball_engagement_near_distance,
    1e-6,
  )
  engagement = (
    (ball_engagement_far_distance - ball_dist) / engagement_span
  ).clamp(min=0.0, max=1.0)
  return ball_dist, engagement


# ---------------------------------------------------------------------------
# Obstacle-aware (dribbling-specific) relaxed rewards
# ---------------------------------------------------------------------------


def _obstacle_relax_gate(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  ball_vel_command_name: str = "ball_vel",
  direction_detection_range: float = 1.5,
  direction_tube_radius: float = 0.5,
  ball_engagement_near_distance: float = 0.3,
  ball_engagement_far_distance: float = 0.75,
  camera_name: str = "robot/d455_color",
  camera_fovy: float = 60.0,
  camera_aspect_ratio: float = 4.0 / 3.0,
  depth_clip: float = 6.0,
) -> torch.Tensor:
  """Return a [0, 1] gate indicating when nominal tracking should be relaxed."""
  try:
    term: ObstacleCommand = env.command_manager.get_term(command_name)
  except Exception:
    return torch.zeros(env.num_envs, device=env.device)

  if term.cfg.num_active == 0:
    return torch.zeros(env.num_envs, device=env.device)

  _, nearest_obs_xy, _ = _get_closest_robot_obstacle(env, command_name)
  in_fov = _camera_fov_mask(env, nearest_obs_xy, camera_name, camera_fovy, camera_aspect_ratio, depth_clip)

  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  cmd_term = env.command_manager.get_term(ball_vel_command_name)
  target_xy = cmd_term.target_position[:, :2]
  target_vec = target_xy - ball_xy
  target_dist = target_vec.norm(dim=-1)
  target_dir = target_vec / target_dist.unsqueeze(-1).clamp(min=1e-6)

  ball_to_obs = nearest_obs_xy - ball_xy
  obs_forward = (ball_to_obs * target_dir).sum(dim=-1)
  obs_lateral = (
    ball_to_obs - obs_forward.unsqueeze(-1) * target_dir
  ).norm(dim=-1)

  corridor_relevant = (
    (target_dist > 1e-6)
    & (obs_forward > 0.0)
    & (obs_forward < target_dist)
    & (obs_forward <= direction_detection_range)
    & (obs_lateral <= direction_tube_radius)
    & in_fov
  )

  _, engagement = _ball_engagement_gate(
    env, ball_engagement_near_distance, ball_engagement_far_distance
  )
  return engagement * corridor_relevant.float()


def ball_vel_tracking_relaxed(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
  obstacle_command_name: str = "adversary",
  direction_detection_range: float = 1.5,
  direction_tube_radius: float = 0.5,
  ball_engagement_near_distance: float = 0.3,
  ball_engagement_far_distance: float = 0.75,
  relax_min_scale: float = 0.2,
  min_speed: float = 0.05,
  camera_name: str = "robot/d455_color",
  camera_fovy: float = 60.0,
  camera_aspect_ratio: float = 4.0 / 3.0,
  depth_clip: float = 6.0,
) -> torch.Tensor:
  """Relax vector tracking only near a relevant in-FOV blocking obstacle."""
  base_reward = ball_vel_tracking_body(env, command_name, sharpness, min_speed)
  relax_gate = _obstacle_relax_gate(
    env,
    command_name=obstacle_command_name,
    ball_vel_command_name=command_name,
    direction_detection_range=direction_detection_range,
    direction_tube_radius=direction_tube_radius,
    ball_engagement_near_distance=ball_engagement_near_distance,
    ball_engagement_far_distance=ball_engagement_far_distance,
    camera_name=camera_name,
    camera_fovy=camera_fovy,
    camera_aspect_ratio=camera_aspect_ratio,
    depth_clip=depth_clip,
  )
  scale = 1.0 - (1.0 - relax_min_scale) * relax_gate
  return scale * base_reward


def ball_vel_norm_relaxed(
  env: ManagerBasedRlEnv,
  command_name: str,
  sharpness: float = 1.0,
  obstacle_command_name: str = "adversary",
  direction_detection_range: float = 1.5,
  direction_tube_radius: float = 0.5,
  ball_engagement_near_distance: float = 0.3,
  ball_engagement_far_distance: float = 0.75,
  relax_min_scale: float = 0.5,
  min_speed: float = 0.05,
  camera_name: str = "robot/d455_color",
  camera_fovy: float = 60.0,
  camera_aspect_ratio: float = 4.0 / 3.0,
  depth_clip: float = 6.0,
) -> torch.Tensor:
  """Relax speed tracking only near a relevant in-FOV blocking obstacle."""
  base_reward = ball_vel_norm(env, command_name, sharpness, min_speed)
  relax_gate = _obstacle_relax_gate(
    env,
    command_name=obstacle_command_name,
    ball_vel_command_name=command_name,
    direction_detection_range=direction_detection_range,
    direction_tube_radius=direction_tube_radius,
    ball_engagement_near_distance=ball_engagement_near_distance,
    ball_engagement_far_distance=ball_engagement_far_distance,
    camera_name=camera_name,
    camera_fovy=camera_fovy,
    camera_aspect_ratio=camera_aspect_ratio,
    depth_clip=depth_clip,
  )
  scale = 1.0 - (1.0 - relax_min_scale) * relax_gate
  return scale * base_reward


def ball_vel_angle_relaxed(
  env: ManagerBasedRlEnv,
  command_name: str,
  obstacle_command_name: str = "adversary",
  direction_detection_range: float = 1.5,
  direction_tube_radius: float = 0.5,
  ball_engagement_near_distance: float = 0.3,
  ball_engagement_far_distance: float = 0.75,
  relax_min_scale: float = 0.2,
  min_speed: float = 0.05,
  camera_name: str = "robot/d455_color",
  camera_fovy: float = 60.0,
  camera_aspect_ratio: float = 4.0 / 3.0,
  depth_clip: float = 6.0,
) -> torch.Tensor:
  """Relax direction tracking only near a relevant in-FOV blocking obstacle."""
  base_reward = ball_vel_angle_body(env, command_name, min_speed)
  relax_gate = _obstacle_relax_gate(
    env,
    command_name=obstacle_command_name,
    ball_vel_command_name=command_name,
    direction_detection_range=direction_detection_range,
    direction_tube_radius=direction_tube_radius,
    ball_engagement_near_distance=ball_engagement_near_distance,
    ball_engagement_far_distance=ball_engagement_far_distance,
    camera_name=camera_name,
    camera_fovy=camera_fovy,
    camera_aspect_ratio=camera_aspect_ratio,
    depth_clip=depth_clip,
  )
  scale = 1.0 - (1.0 - relax_min_scale) * relax_gate
  return scale * base_reward


# ---------------------------------------------------------------------------
# Camera projections (training observations)
# ---------------------------------------------------------------------------


def _project_ball_to_camera(
  env: ManagerBasedRlEnv,
  camera_name: str,
  aspect_ratio: float,
  head_camera_fovy: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Project ball world position into normalised camera image coordinates."""
  cam_id = env.sim.mj_model.camera(camera_name).id
  cam_pos = env.sim.data.cam_xpos[:, cam_id, :]
  cam_mat = env.sim.data.cam_xmat[:, cam_id, :].reshape(-1, 3, 3)

  ball_pos = env.scene["ball"].data.root_link_pos_w
  p_rel = ball_pos - cam_pos
  p_cam = torch.bmm(cam_mat.transpose(1, 2), p_rel.unsqueeze(-1)).squeeze(-1)

  depth = (-p_cam[:, 2]).clamp_min(1e-6)
  in_front = p_cam[:, 2] < 0

  tx = p_cam[:, 0] / depth
  ty = p_cam[:, 1] / depth

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
  """Normalised ball position in camera image frame — observation term. Shape [N, 2]."""
  nx, ny, in_front = _project_ball_to_camera(env, camera_name, aspect_ratio)
  nx = torch.where(in_front, nx, nx.clamp(-2.0, 2.0))
  ny = torch.where(in_front, ny, ny.clamp(-2.0, 2.0))
  return torch.stack([nx, ny], dim=-1)


def head_ball_tracking(
  env: ManagerBasedRlEnv,
  camera_name: str = "robot/d455_color",
) -> torch.Tensor:
  """Cosine similarity between camera optical axis and camera→ball direction."""
  cam_id = env.sim.mj_model.camera(camera_name).id
  cam_pos = env.sim.data.cam_xpos[:, cam_id, :]
  cam_mat = env.sim.data.cam_xmat[:, cam_id, :].reshape(-1, 3, 3)

  ball_pos = env.scene["ball"].data.root_link_pos_w
  to_ball = ball_pos - cam_pos
  to_ball = to_ball / to_ball.norm(dim=-1, keepdim=True).clamp(min=1e-6)

  cam_fwd = -cam_mat[:, :, 2]
  cos_sim = (cam_fwd * to_ball).sum(dim=-1).clamp(-1.0, 1.0)
  return (1.0 + cos_sim) / 2.0


# ---------------------------------------------------------------------------
# Target rewards
# ---------------------------------------------------------------------------


def ball_target_reached(
  env: ManagerBasedRlEnv,
  command_name: str = "ball_vel",
) -> torch.Tensor:
  """Discrete bonus fired once when the ball crosses the target threshold."""
  from colosseum.tasks.dribbling.mdp.ball_velocity_command import BallVelocityCommand

  term: BallVelocityCommand = env.command_manager.get_term(command_name)
  return term.target_reached_mask.float()


def ball_target_progress(
  env: ManagerBasedRlEnv,
  command_name: str = "ball_vel",
  obstacle_command_name: str = "adversary",
  target_near_distance: float = 0.4,
  target_far_distance: float = 1.0,
  target_obstacle_near_distance: float = 0.6,
  target_obstacle_far_distance: float = 1.2,
  speed_ref: float = 1.0,
  distance_scale_ref: float = 2.0,
  distance_scale_max: float = 1.5,
  camera_name: str = "robot/d455_color",
  camera_fovy: float = 60.0,
  camera_aspect_ratio: float = 4.0 / 3.0,
  depth_clip: float = 6.0,
) -> torch.Tensor:
  """Reward ball velocity toward the persistent target, gated off near the target."""
  cmd_term = env.command_manager.get_term(command_name)
  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  ball_vel_xy = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  target_xy = cmd_term.target_position[:, :2]

  target_vec = target_xy - ball_xy
  target_dist = target_vec.norm(dim=-1)
  target_dir = target_vec / target_dist.unsqueeze(-1).clamp(min=1e-6)

  progress_speed = (ball_vel_xy * target_dir).sum(dim=-1).clamp(min=0.0)
  progress_reward = (progress_speed / max(speed_ref, 1e-6)).clamp(min=0.0, max=1.0)

  gate_span = max(target_far_distance - target_near_distance, 1e-6)
  target_gate = (
    (target_dist - target_near_distance) / gate_span
  ).clamp(min=0.0, max=1.0)

  distance_scale = 1.0 + (distance_scale_max - 1.0) * (
    target_dist / max(distance_scale_ref, 1e-6)
  ).clamp(min=0.0, max=1.0)

  obstacle_gate = torch.ones(env.num_envs, device=env.device)
  try:
    term: ObstacleCommand = env.command_manager.get_term(obstacle_command_name)
    if term.cfg.num_active > 0:
      obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]
      target_obs_dists = (obs_xy - target_xy.unsqueeze(1)).norm(dim=-1)
      min_idx = target_obs_dists.argmin(dim=-1)
      nearest_to_target_xy = obs_xy[torch.arange(env.num_envs, device=env.device), min_idx]
      target_obs_dist = target_obs_dists.min(dim=-1).values
      in_fov = _camera_fov_mask(env, nearest_to_target_xy, camera_name, camera_fovy, camera_aspect_ratio, depth_clip)
      obstacle_gate_span = max(
        target_obstacle_far_distance - target_obstacle_near_distance,
        1e-6,
      )
      raw_gate = (
        (target_obs_dist - target_obstacle_near_distance) / obstacle_gate_span
      ).clamp(min=0.0, max=1.0)
      obstacle_gate = torch.where(in_fov, raw_gate, torch.ones_like(raw_gate))
  except Exception:
    pass

  return target_gate * obstacle_gate * distance_scale * progress_reward


# ---------------------------------------------------------------------------
# Obstacle penalty rewards
# ---------------------------------------------------------------------------


def robot_obstacle_collision(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  collision_near_distance: float = 0.5,
  collision_far_distance: float = 1.5,
) -> torch.Tensor:
  """Local body-obstacle collision penalty."""
  term: ObstacleCommand = env.command_manager.get_term(command_name)
  if term.cfg.num_active == 0:
    return torch.zeros(env.num_envs, device=env.device)

  min_dist, nearest_obs_xy, _ = _get_closest_robot_obstacle(env, command_name)
  del nearest_obs_xy

  collision_span = max(collision_far_distance - collision_near_distance, 1e-6)
  collision_progress = (
    (collision_far_distance - min_dist) / collision_span
  ).clamp(min=0.0, max=1.0)
  return collision_progress.pow(2)


def ball_obstacle_collision(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  collision_near_distance: float = 0.15,
  collision_far_distance: float = 0.6,
) -> torch.Tensor:
  """Ball-obstacle proximity penalty."""
  term: ObstacleCommand = env.command_manager.get_term(command_name)
  if term.cfg.num_active == 0:
    return torch.zeros(env.num_envs, device=env.device)

  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  obs_xy = term.obstacle_positions_w[:, : term.cfg.num_active]
  dist = (obs_xy - ball_xy.unsqueeze(1)).norm(dim=-1)
  min_dist = dist.min(dim=-1).values

  collision_span = max(collision_far_distance - collision_near_distance, 1e-6)
  collision_progress = (
    (collision_far_distance - min_dist) / collision_span
  ).clamp(min=0.0, max=1.0)
  return collision_progress.pow(2)


def obstacle_direction(
  env: ManagerBasedRlEnv,
  command_name: str = "adversary",
  ball_vel_command_name: str = "ball_vel",
  direction_detection_range: float = 1.5,
  direction_tube_radius: float = 0.5,
  direction_sharpness: float = 3.0,
  min_ball_speed: float = 0.1,
  ball_engagement_near_distance: float = 0.3,
  ball_engagement_far_distance: float = 0.75,
  camera_name: str = "robot/d455_color",
  camera_fovy: float = 60.0,
  camera_aspect_ratio: float = 4.0 / 3.0,
  depth_clip: float = 6.0,
) -> torch.Tensor:
  """Penalty for the ball actually moving toward an obstacle on the ball-target segment."""
  term: ObstacleCommand = env.command_manager.get_term(command_name)
  if term.cfg.num_active == 0:
    return torch.zeros(env.num_envs, device=env.device)

  _, nearest_obs_xy, _ = _get_closest_robot_obstacle(env, command_name)
  in_fov = _camera_fov_mask(env, nearest_obs_xy, camera_name, camera_fovy, camera_aspect_ratio, depth_clip)

  ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
  ball_vel_xy = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
  cmd_term = env.command_manager.get_term(ball_vel_command_name)
  target_xy = cmd_term.target_position[:, :2]
  target_vec = target_xy - ball_xy
  target_dist = target_vec.norm(dim=-1)
  target_dir = target_vec / target_dist.unsqueeze(-1).clamp(min=1e-6)

  ball_to_obs = nearest_obs_xy - ball_xy
  obs_forward = (ball_to_obs * target_dir).sum(dim=-1)
  obs_lateral = (
    ball_to_obs - obs_forward.unsqueeze(-1) * target_dir
  ).norm(dim=-1)

  ball_speed = ball_vel_xy.norm(dim=-1)
  direction_relevant = (
    (target_dist > 1e-6)
    & (obs_forward > 0.0)
    & (obs_forward < target_dist)
    & (obs_forward <= direction_detection_range)
    & (obs_lateral <= direction_tube_radius)
    & (ball_speed > min_ball_speed)
    & in_fov
  )

  ball_vel_dir = ball_vel_xy / ball_speed.unsqueeze(-1).clamp(min=1e-6)
  ball_to_obs_dir = ball_to_obs / ball_to_obs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  alignment = (ball_vel_dir * ball_to_obs_dir).sum(dim=-1).clamp(-1.0, 1.0)
  toward_obstacle = alignment.clamp(min=0.0)
  direction_term = toward_obstacle.pow(2)
  if direction_sharpness != 2.0:
    direction_term = direction_term.pow(direction_sharpness / 2.0)

  _, engagement = _ball_engagement_gate(
    env, ball_engagement_near_distance, ball_engagement_far_distance
  )
  return engagement * direction_relevant.float() * direction_term
