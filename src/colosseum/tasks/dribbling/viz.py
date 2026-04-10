"""Debug visualization helpers for the dribbling task."""

from __future__ import annotations

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


def draw_camera_ball_overlay(env, vis: DebugVisualizer) -> None:
  """Draw head camera frame and arrow to ball in the mjlab viewer.

  Arrow colour indicates the current Phase 2 encoding mode:
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
  # cam_xmat columns are camera local axes in world frame:
  #   col 0 = x (right), col 1 = y (up), col 2 = z (backward)
  # Convert to standard pinhole convention: x=right, y=down, z=forward
  cam_mat = env.sim.data.cam_xmat[env_idx, cam_id, :].cpu().numpy().reshape(3, 3)
  optical_mat = cam_mat.copy()
  optical_mat[:, 1] *= -1  # y: up → down
  optical_mat[:, 2] *= -1  # z: backward → forward
  ball_pos = ball.data.root_link_pos_w[env_idx, :].cpu().numpy()

  # Draw a small arrow pointing towards the ball
  direction = ball_pos - cam_pos
  distance = np.linalg.norm(direction)
  arrow_end = cam_pos + (direction / distance) * 0.1 if distance > 0.1 else ball_pos

  # Colour from FOV state: green = depth encoder active, red = fallback, yellow = N/A
  ball_term = _get_ball_term(env)
  if ball_term is not None and ball_term._ball_in_fov is not None:
    adapt_mask = ball_term.get_adaptation_mask()
    valid = adapt_mask is not None and adapt_mask[env_idx].item()
    color = (0.1, 0.85, 0.1, 0.9) if valid else (0.9, 0.1, 0.1, 0.9)
  else:
    color = (1.0, 0.85, 0.0, 0.8)

  vis.add_frame(position=cam_pos, rotation_matrix=optical_mat, scale=0.1, axis_radius=0.005, alpha=0.5)
  vis.add_arrow(start=cam_pos, end=arrow_end, color=color, width=0.005)


class DribblingViz:
  """Viz callback: draws head camera frame + colour-coded arrow to ball.

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
