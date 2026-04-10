"""Debug visualization helpers for the dribbling task."""

from __future__ import annotations

import math

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _get_ball_term(env):
  """Safely return the ball RMA term, or None if not present."""
  try:
    unwrapped = getattr(env, "unwrapped", env)
    return unwrapped.rma_manager._terms.get("ball")
  except AttributeError:
    return None


def _draw_fov_frustum(
  vis: DebugVisualizer,
  cam_pos: np.ndarray,
  cam_mat: np.ndarray,
  color: tuple,
  fovy_deg: float = 60.0,
  aspect: float = 4.0 / 3.0,
  depth: float = 2.0,
) -> None:
  """Draw a camera FOV frustum as 8 cylinder edges.

  MuJoCo camera convention: optical axis = -z, x = right, y = up.
  Far-plane corners in camera frame: (±hw, ±hv, -depth).
  World-frame corners: cam_pos + cam_mat @ p_cam_corner.

  Args:
    vis:      DebugVisualizer instance.
    cam_pos:  Camera origin in world frame, shape (3,).
    cam_mat:  Camera rotation matrix (columns = local axes in world), shape (3,3).
    color:    RGBA tuple for all edges.
    fovy_deg: Vertical FOV in degrees.
    aspect:   Width/height pixel ratio (horizontal FOV derived from this).
    depth:    Far-plane distance in metres.
  """
  tan_half_v = math.tan(math.radians(fovy_deg / 2))
  tan_half_h = tan_half_v * aspect
  hw = tan_half_h * depth
  hv = tan_half_v * depth

  # Corners in camera frame (ordered: TR, TL, BL, BR)
  corners_cam = np.array([
    [ hw,  hv, -depth],  # top-right
    [-hw,  hv, -depth],  # top-left
    [-hw, -hv, -depth],  # bottom-left
    [ hw, -hv, -depth],  # bottom-right
  ])

  # Convert to world frame
  corners_w = [cam_pos + cam_mat @ c for c in corners_cam]

  # 4 apex → corner edges
  for corner in corners_w:
    vis.add_cylinder(start=cam_pos, end=corner, radius=0.003, color=color)

  # 4 far-plane edges (rectangle outline)
  for i in range(4):
    vis.add_cylinder(
      start=corners_w[i],
      end=corners_w[(i + 1) % 4],
      radius=0.003,
      color=color,
    )


def draw_camera_ball_overlay(env, vis: DebugVisualizer) -> None:
  """Draw head camera FOV frustum and line-to-ball in the mjlab viewer.

  Frustum and line colour indicates the current Phase 2 encoding mode:
    Green  — ball is within camera FOV and depth buffer is warmed up
             (depth encoder active, adaptation signal valid).
    Red    — ball is outside FOV or buffer is still warming up
             (privileged-encoder fallback active).
    Yellow — no FOV tracking available (Phase 1 or play without camera).

  No-ops silently if the d455_color camera or ball entity are absent.
  """
  try:
    cam_name = "robot/d455_color"
    cam_id = env.sim.mj_model.camera(cam_name).id
    ball = env.scene["ball"]
  except (KeyError, Exception):
    return

  env_idx = vis.env_idx
  cam_pos = env.sim.data.cam_xpos[env_idx, cam_id, :].cpu().numpy()
  cam_mat = env.sim.data.cam_xmat[env_idx, cam_id, :].cpu().numpy().reshape(3, 3)
  # cam_xmat columns: col 0 = x (right), col 1 = y (up), col 2 = z (backward)
  # Convert to standard pinhole convention: x=right, y=down, z=forward
  optical_mat = cam_mat.copy()
  optical_mat[:, 1] *= -1  # y: up → down
  optical_mat[:, 2] *= -1  # z: backward → forward
  ball_pos = ball.data.root_link_pos_w[env_idx, :].cpu().numpy()

  # Determine colour from FOV / adaptation-mask state
  ball_term = _get_ball_term(env)
  if ball_term is not None and ball_term._ball_in_fov is not None:
    adapt_mask = ball_term.get_adaptation_mask()
    valid = adapt_mask is not None and adapt_mask[env_idx].item()
    line_color = (0.1, 0.85, 0.1, 0.9) if valid else (0.9, 0.1, 0.1, 0.9)
    frustum_color = (0.1, 0.85, 0.1, 0.5) if valid else (0.9, 0.1, 0.1, 0.5)
  else:
    line_color = (1.0, 0.85, 0.0, 0.8)
    frustum_color = (1.0, 0.85, 0.0, 0.4)

  # Camera frame axes
  vis.add_frame(
    position=cam_pos,
    rotation_matrix=optical_mat,
    scale=0.3,
    axis_radius=0.005,
    alpha=0.5,
  )

  # Line from camera to ball (full distance so relative depth is visible)
  vis.add_cylinder(start=cam_pos, end=ball_pos, radius=0.004, color=line_color)
  # Small sphere at ball position for emphasis
  vis.add_sphere(center=ball_pos, radius=0.03, color=(*line_color[:3], 0.6))

  # FOV frustum
  fovy = ball_term.cfg.camera_fovy if ball_term is not None else 60.0
  aspect = ball_term.cfg.camera_aspect_ratio if ball_term is not None else 4.0 / 3.0
  _draw_fov_frustum(
    vis, cam_pos, cam_mat, frustum_color, fovy_deg=fovy, aspect=aspect, depth=2.0
  )


class DribblingViz:
  """Viz callback: draws head camera FOV frustum + line to ball.

  Green  → depth encoder active (ball in FOV, buffer warm).
  Red    → privileged fallback active (ball out of FOV or warming up).
  Yellow → no FOV tracking (Phase 1 or play without depth camera).

  Registered via RmaBasedEnvCfg.viz_callbacks so DribblingEnv is not needed.
  factory(env) → DribblingViz instance with debug_vis(vis).
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env

  def debug_vis(self, vis) -> None:
    draw_camera_ball_overlay(self._env, vis)
