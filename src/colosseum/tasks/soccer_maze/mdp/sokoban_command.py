"""Velocity command produced by the Sokoban discrete plan.

Signals
  ball_vel    [N, 2]  PUSH only: push_dir × ball_speed  (world frame)
  robot_vel   [N, 2]  MOVE: toward next grid cell; PUSH: toward the ball (body frame)
  omega_z     [N]     heading P-controller (both phases)
  is_push     [N]     True during PUSH actions
  heading_cos [N]     X-component of target direction in body frame (both phases)

During MOVE the robot walks toward the next Sokoban grid cell.  During PUSH
the robot walks toward the ball, letting it approach from whatever angle is
needed to execute the kick; the robot–ball relationship rewards
(``robot_ball_yaw``, ``robot_ball_approach_vel_push``) provide finer guidance.

.command returns [vx_body, vy_body, omega_z] — non-zero in both phases, so
generic mjlab rewards (feet_swing_height, soft_landing, action_rate) remain
active throughout.  Task-specific ball rewards read ``ball_vel`` (world frame)
directly via ``get_term(...)``.

Direction conversion
  The Sokoban plan stores directions in grid space (di, dj).
  Grid-to-local: dx = dj, dy = -di  (grid i downward, local y upward).
  The local-frame unit direction is EMA-smoothed to avoid abrupt jumps at plan
  step boundaries, then rotated into body frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply

from colosseum.mdp.abstraction.maze.sokoban_grid_abstraction import (
  SokobanGridAbstraction,
)
from colosseum.managers.abstraction_manager import AbstractionManager

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class SokobanCommand(CommandTerm):
  """Velocity command driven by the current Sokoban plan step."""

  cfg: SokobanCommandCfg

  def __init__(self, cfg: SokobanCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    assert hasattr(env, "abstraction_manager") and isinstance(
      env.abstraction_manager, AbstractionManager
    ), "Requires an env with an abstraction_manager (AbstractionBasedEnv or ConstraintAbstractionBasedEnv)"
    self.env = env

    self._ball_vel = torch.zeros((env.num_envs, 2), device=env.device)
    self._robot_vel = torch.zeros((env.num_envs, 2), device=env.device)
    self._omega_z = torch.zeros(env.num_envs, device=env.device)
    self._is_push = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._heading_cos = torch.zeros(env.num_envs, device=env.device)
    self._push_target_pos = torch.zeros((env.num_envs, 2), device=env.device)

    # Previous smoothed local-frame direction (for EMA smoothing).
    self._prev_dir_local = torch.zeros((env.num_envs, 2), device=env.device)

    # Ball position captured at the start of each PUSH action.
    self._push_start_ball_pos = torch.zeros((env.num_envs, 2), device=env.device)
    self._prev_is_push = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    self.metrics["ball_speed"] = torch.zeros(env.num_envs, device=env.device)
    self.metrics["robot_speed"] = torch.zeros(env.num_envs, device=env.device)

  @property
  def command(self) -> torch.Tensor:
    # Shape [N, 3]: [vx_body, vy_body, omega_z] — MOVE locomotion, zero in PUSH.
    # Generic mjlab rewards (feet_swing_height, soft_landing, action_rate) read
    # this.  Task-specific ball rewards read `ball_vel` directly via `get_term()`.
    return torch.cat([self._robot_vel, self._omega_z.unsqueeze(-1)], dim=-1)

  # ── Sub-command accessors ────────────────────────────────────────────────────

  @property
  def ball_vel(self) -> torch.Tensor:
    """[N, 2] ball world-frame velocity command; zero outside PUSH."""
    return self._ball_vel

  @property
  def robot_lin_vel(self) -> torch.Tensor:
    """[N, 2] robot body-frame velocity command; MOVE only, zero during PUSH."""
    return self._robot_vel

  @property
  def robot_omega_z(self) -> torch.Tensor:
    """[N] robot body-frame yaw-rate command; MOVE only, zero during PUSH."""
    return self._omega_z

  @property
  def is_push(self) -> torch.Tensor:
    """[N] bool: True when the current plan action is a PUSH."""
    return self._is_push

  @property
  def heading_cos(self) -> torch.Tensor:
    """[N] cos(heading_error) during MOVE; zero during PUSH."""
    return self._heading_cos

  @property
  def push_start_ball_pos(self) -> torch.Tensor:
    """[N, 2] ball world-frame position captured at the start of each PUSH."""
    return self._push_start_ball_pos

  @property
  def push_target_pos(self) -> torch.Tensor:
    """[N, 2] world-frame center of the ball's target cell; valid during PUSH only."""
    return self._push_target_pos

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._ball_vel[env_ids] = 0.0
    self._robot_vel[env_ids] = 0.0
    self._omega_z[env_ids] = 0.0
    self._is_push[env_ids] = False
    self._heading_cos[env_ids] = 0.0
    self._prev_dir_local[env_ids] = 0.0
    self._push_start_ball_pos[env_ids] = 0.0
    self._prev_is_push[env_ids] = False
    self._push_target_pos[env_ids] = 0.0

  def _update_command(self) -> None:
    abstraction = self.env.abstraction_manager.get_term(self.cfg.abstraction_name)
    assert isinstance(abstraction, SokobanGridAbstraction)

    # ── Active mask and action type ──────────────────────────────────────────
    active = abstraction.current_step < abstraction.plan_length  # [N]
    is_push = abstraction._current_action_is_push & active  # [N]
    is_move = (~abstraction._current_action_is_push) & active  # [N]

    # ── Grid direction → local (world-like) frame ────────────────────────────
    # Grid: di=row-delta (down), dj=col-delta (right).  Local: x = dj, y = -di.
    dir_grid = abstraction._current_direction_grid  # [N, 2]: (di, dj)
    dir_local = torch.stack([dir_grid[:, 1], -dir_grid[:, 0]], dim=1)
    raw_norm = dir_local.norm(dim=1, keepdim=True)
    dir_local_unit = dir_local / raw_norm.clamp(min=1e-6)

    # ── EMA smoothing on the local direction ─────────────────────────────────
    # Prevents abrupt 90° jumps at plan step boundaries (the previous "brake at
    # every new cell" symptom).  For envs with no direction yet (|raw|==0 on the
    # first active step) we snap to dir_local_unit instead of smoothing from zero.
    alpha = self.cfg.ema_smoothing
    prev_has_dir = self._prev_dir_local.norm(dim=-1, keepdim=True) > 1e-6
    smoothed = torch.where(
      prev_has_dir,
      alpha * dir_local_unit + (1.0 - alpha) * self._prev_dir_local,
      dir_local_unit,
    )
    smoothed_unit = smoothed / smoothed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    self._prev_dir_local = smoothed_unit.clone()

    # ── Ball velocity (PUSH only, world frame) ───────────────────────────────
    self._ball_vel[:] = (
      smoothed_unit * self.cfg.ball_speed * is_push.float().unsqueeze(1)
    )

    # ── Shared rotation to body frame ────────────────────────────────────────
    root_quat_w = self.env.scene["robot"].data.root_link_quat_w
    quat_conj = torch.cat([root_quat_w[:, :1], -root_quat_w[:, 1:]], dim=-1)

    fwd = torch.tensor(
      self.cfg.body_forward_axis[:2], device=self.env.device, dtype=torch.float32
    )

    def _body_dir_and_heading(
      local_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      """Rotate a local-frame 2-D unit direction into body frame; return (dir_b, heading_err)."""
      local_3d = torch.cat(
        [local_2d, torch.zeros(self.env.num_envs, 1, device=self.env.device)], dim=-1
      )
      d_b = quat_apply(quat_conj, local_3d)[:, :2]
      d_b = d_b / d_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
      h_err = torch.atan2(
        fwd[0] * d_b[:, 1] - fwd[1] * d_b[:, 0],
        fwd[0] * d_b[:, 0] + fwd[1] * d_b[:, 1],
      )
      return d_b, h_err

    # ── MOVE: walk toward the planned grid cell ───────────────────────────────
    dir_move_b, heading_err_move = _body_dir_and_heading(smoothed_unit)
    alignment_move = (
      self.cfg.min_alignment_scale
      + (1.0 - self.cfg.min_alignment_scale) * heading_err_move.abs().cos()
    )
    speed_move = (self.cfg.robot_speed * alignment_move).clamp(
      self.cfg.min_velocity, self.cfg.max_velocity
    )

    # ── PUSH: walk in push direction at controlled speed ──────────────────────
    # Same direction as the ball push (smoothed_unit); speed is intentionally
    # lower than MOVE to avoid over-shooting and corrupting ball contact.
    dir_push_b = dir_move_b
    heading_err_push = heading_err_move
    alignment_push = (
      self.cfg.min_alignment_scale
      + (1.0 - self.cfg.min_alignment_scale) * heading_err_push.abs().cos()
    )
    speed_push = (self.cfg.push_robot_speed * alignment_push).clamp(
      self.cfg.min_velocity, self.cfg.push_robot_speed
    )

    # ── Combine: mutually exclusive MOVE / PUSH ───────────────────────────────
    move_f = is_move.float()
    push_f = is_push.float()
    self._robot_vel[:] = dir_move_b * speed_move.unsqueeze(-1) * move_f.unsqueeze(
      -1
    ) + dir_push_b * speed_push.unsqueeze(-1) * push_f.unsqueeze(-1)
    self._omega_z[:] = (heading_err_move * self.cfg.angular_velocity_gain).clamp(
      -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity
    ) * move_f + (heading_err_push * self.cfg.angular_velocity_gain).clamp(
      -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity
    ) * push_f
    self._heading_cos[:] = dir_move_b[:, 0] * move_f + dir_push_b[:, 0] * push_f
    # Capture ball position at MOVE→PUSH transition.
    just_started_push = is_push & ~self._prev_is_push
    ball_pos_w = self.env.scene["ball"].data.root_link_pos_w[:, :2]
    self._push_start_ball_pos = torch.where(
      just_started_push.unsqueeze(-1),
      ball_pos_w,
      self._push_start_ball_pos,
    )
    self._prev_is_push = is_push.clone()
    self._is_push[:] = is_push

    # Compute world-frame center of the ball's target cell for PUSH reward.
    target_local = abstraction._grid_to_local(abstraction.expected_next_ball_cell, center=True)
    target_world = target_local + self.env.scene.env_origins[:, :2]
    self._push_target_pos = torch.where(
      is_push.unsqueeze(-1), target_world, self._push_target_pos
    )

  def _update_metrics(self) -> None:
    self.metrics["ball_speed"] = self._ball_vel.norm(dim=-1)
    self.metrics["robot_speed"] = self._robot_vel.norm(dim=-1)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    robot = self.env.scene["robot"]
    origin = self.env.scene.env_origins[batch, :2]
    robot_pos_local = robot.data.root_link_pos_w[batch, :2] - origin
    ball_pos_local = self.env.scene["ball"].data.root_link_pos_w[batch, :2] - origin

    # Robot walking target (body → world), always drawn.
    rv_body = self._robot_vel[batch]
    rv_3d = torch.cat([rv_body, torch.zeros(1, device=self.env.device)]).unsqueeze(0)
    rv_world = quat_apply(robot.data.root_link_quat_w[batch].unsqueeze(0), rv_3d)
    rv_np = rv_world.squeeze(0).cpu().numpy()
    tag = "push" if self._is_push[batch].item() else "move"
    start_r = np.array([robot_pos_local[0].item(), robot_pos_local[1].item(), 0.3])
    visualizer.add_arrow(
      start=start_r,
      end=start_r + rv_np * 2.0,
      color=(0.2, 0.8, 1.0, 0.9),
      label=f"{tag} ω={self._omega_z[batch].item():.2f}",
    )

    # Ball target (world), only drawn during PUSH.
    if self._is_push[batch].item():
      bv = self._ball_vel[batch].cpu().numpy()
      start_b = np.array([ball_pos_local[0].item(), ball_pos_local[1].item(), 0.2])
      visualizer.add_arrow(
        start=start_b,
        end=start_b + np.array([bv[0], bv[1], 0.0]) * 2.0,
        color=(1.0, 0.5, 0.0, 0.9),
        label=f"ball |v|={float(np.linalg.norm(bv)):.2f}",
      )


@dataclass(kw_only=True)
class SokobanCommandCfg(CommandTermCfg):
  """Configuration for SokobanCommand."""

  class_type: type[CommandTerm] = SokobanCommand
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  debug_vis: bool = True

  abstraction_name: str = "sokoban"

  ball_speed: float = 0.5
  robot_speed: float = 1.0
  push_robot_speed: float = 0.3  # slower during PUSH to avoid corrupting ball contact

  body_forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
  min_velocity: float = 0.1
  max_velocity: float = 2.0
  min_alignment_scale: float = 0.3
  angular_velocity_gain: float = 2.0
  max_angular_velocity: float = 2.0

  # EMA smoothing on the plan direction.  Mirrors AbstractionVelocityCommand:
  # low alpha = heavy smoothing across plan-step boundaries, preventing the
  # robot from braking each time the Sokoban plan advances to a new direction.
  ema_smoothing: float = 0.1

  def build(self, env: ManagerBasedRlEnv) -> SokobanCommand:
    return SokobanCommand(self, env)
