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
  # GT marker raised above ball mesh so it's always visible
  gt_marker_pos = ball_pos.copy()
  gt_marker_pos[2] += 0.15  # above ball mesh (radius=0.11)
  vis.add_sphere(center=gt_marker_pos, radius=0.03, color=(*line_color[:3], 0.6))

  # FOV frustum
  fovy = ball_term.cfg.camera_fovy if ball_term is not None else 60.0
  aspect = ball_term.cfg.camera_aspect_ratio if ball_term is not None else 4.0 / 3.0
  _draw_fov_frustum(
    vis, cam_pos, cam_mat, frustum_color, fovy_deg=fovy, aspect=aspect, depth=2.0
  )


def draw_ball_prediction_overlay(env, vis: DebugVisualizer) -> None:
  """Draw predicted ball position (from depth encoder) as a blue sphere.

  Compares the BallHead prediction against the GT ball position.
  Only active when the depth encoder is running (Phase 2 with ball in FOV).

  Blue sphere  = predicted ball position (from depth encoder + ball head)
  Red sphere   = GT ball position (already drawn by draw_camera_ball_overlay)
  Cyan line    = connects predicted to GT position (shows prediction error)
  """
  ball_term = _get_ball_term(env)
  if ball_term is None:
    return

  pred = ball_term.predict_ball_state()
  if pred is None:
    return

  env_idx = vis.env_idx
  adapt_mask = ball_term.get_adaptation_mask()
  if adapt_mask is None or not adapt_mask[env_idx].item():
    return  # Only show when depth encoder is active

  # pred is [x, y, vx, vy] in robot body frame (same as privileged_ball)
  pred_xy = pred[env_idx, :2].cpu().numpy()  # (2,)

  # Convert body-frame XY prediction back to world frame for visualization
  import torch

  robot = env.scene["robot"]
  robot_pos_w = robot.data.root_link_pos_w[env_idx, :3].cpu().numpy()
  robot_quat_w = robot.data.root_link_quat_w[env_idx, :].cpu()

  # Rotate body-frame offset to world frame: q * [x, y, 0]
  from mjlab.utils.lab_api.math import quat_apply

  offset_b = torch.tensor([pred_xy[0], pred_xy[1], 0.0], dtype=torch.float32)
  offset_w = quat_apply(robot_quat_w.unsqueeze(0), offset_b.unsqueeze(0)).squeeze(0).numpy()

  pred_pos_w = robot_pos_w + offset_w
  # Raise above ball mesh so both markers are visible
  pred_pos_w[2] = 0.11 + 0.15  # ball radius + offset

  # Blue sphere at predicted position
  vis.add_sphere(
    center=pred_pos_w,
    radius=0.03,
    color=(0.1, 0.3, 1.0, 0.7),
  )

  # Cyan line from prediction to GT marker (both raised)
  try:
    ball_pos_gt = env.scene["ball"].data.root_link_pos_w[env_idx, :3].cpu().numpy()
    gt_marker_pos = ball_pos_gt.copy()
    gt_marker_pos[2] += 0.15
    vis.add_cylinder(
      start=pred_pos_w,
      end=gt_marker_pos,
      radius=0.003,
      color=(0.0, 0.9, 0.9, 0.6),
    )
  except (KeyError, Exception):
    pass


def draw_depth_window(env, env_idx: int) -> None:
  """Show the encoder's current preprocessed depth frame in a cv2 window.

  The window appears only when current adaptation frames are available
  (Phase 2 with camera in scene) and is a no-op otherwise.
  """
  ball_term = _get_ball_term(env)
  if ball_term is None or ball_term._current_frame is None:
    return

  try:
    import cv2
  except ImportError:
    return

  frame = ball_term._current_frame[env_idx, 0].cpu().numpy()  # (H, W)
  frame_max = frame.max()
  if frame_max > 0:
    frame = frame / frame_max

  tile = cv2.applyColorMap(
    (frame * 255).clip(0, 255).astype("uint8"),
    cv2.COLORMAP_INFERNO,
  )

  win = "Depth (encoder input)"
  cv2.namedWindow(win, cv2.WINDOW_NORMAL)
  cv2.setWindowTitle(win, f"Depth — env {env_idx}")
  cv2.imshow(win, tile)
  cv2.waitKey(1)


class DribblingViz:
  """Viz callback: draws head camera FOV frustum + line to ball + ball prediction.

  3-D overlay (in MuJoCo viewer):
    Green  → depth encoder active (ball in FOV, buffer warm).
    Red    → privileged fallback active (ball out of FOV or warming up).
    Yellow → no FOV tracking (Phase 1 or play without depth camera).
    Blue sphere → ball position predicted by depth encoder's ball head.
    Cyan line   → connects predicted position to GT (shows prediction error).

  Optional cv2 depth window (set show_depth=True to enable):
    Current preprocessed depth frame fed to the encoder.

  Registered via RmaBasedEnvCfg.viz_callbacks so DribblingEnv is not needed.
  factory(env) → DribblingViz instance with debug_vis(vis).
  """

  def __init__(self, env: ManagerBasedRlEnv, *, show_depth: bool = False) -> None:
    self._env = env
    self._show_depth = show_depth

  def debug_vis(self, vis) -> None:
    draw_camera_ball_overlay(self._env, vis)
    draw_ball_prediction_overlay(self._env, vis)
    self._print_ball_estimate(vis.env_idx)
    if self._show_depth:
      draw_depth_window(self._env, vis.env_idx)

  def _print_ball_estimate(self, env_idx: int) -> None:
    """Print estimated vs GT ball position and velocity to terminal."""
    ball_term = _get_ball_term(self._env)
    if ball_term is None:
      return

    pred = ball_term.predict_ball_state()
    if pred is None:
      return

    adapt_mask = ball_term.get_adaptation_mask()
    active = adapt_mask is not None and adapt_mask[env_idx].item()

    p = pred[env_idx].cpu()
    pred_x, pred_y, pred_vx, pred_vy = p[0].item(), p[1].item(), p[2].item(), p[3].item()

    # GT from observation functions (same frame as prediction)
    try:
      gt_tensor = self._env.observation_manager.compute_group("privileged_ball")
      gt = gt_tensor[env_idx].cpu()
      gt_x, gt_y, gt_vx, gt_vy = gt[0].item(), gt[1].item(), gt[2].item(), gt[3].item()
    except Exception:
      gt_x = gt_y = gt_vx = gt_vy = float("nan")

    status = "\033[92mDEPTH\033[0m" if active else "\033[91mGT-FB\033[0m"
    print(
      f"[{status}] "
      f"pos: ({pred_x:+.3f}, {pred_y:+.3f}) vs GT ({gt_x:+.3f}, {gt_y:+.3f})  "
      f"vel: ({pred_vx:+.3f}, {pred_vy:+.3f}) vs GT ({gt_vx:+.3f}, {gt_vy:+.3f})  "
      f"err_pos: {((pred_x-gt_x)**2 + (pred_y-gt_y)**2)**0.5:.4f}  "
      f"err_vel: {((pred_vx-gt_vx)**2 + (pred_vy-gt_vy)**2)**0.5:.4f}",
      end="\r",
    )
