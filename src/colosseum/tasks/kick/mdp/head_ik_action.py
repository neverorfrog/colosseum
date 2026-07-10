"""IK-based head tracking action term for the residual dribbling task.

Ported from the ``dribbling`` task. Computes head yaw/pitch analytically from
the ball position so the camera points at the ball, and writes those targets
over the (frozen-walk + residual) head outputs on every decimation substep.
Consumes no policy output (action_dim=0).

Difference from the ``dribbling`` version: instead of a periodic time-based
scan, the head returns to the neutral (0, 0) pose whenever the ball is out of
view -- i.e. beyond the head's tracking reach (``fov_half``) or past
``max_range``. This mirrors the on-robot behaviour (you can't track what the
camera can't see) and keeps the head's FOV gate consistent with the ball
perception model used for the policy's ball observation.

The point of running this during training is not perception itself but
disturbance: the head physically swivels and snaps back, perturbing the IMU and
base dynamics, so the balance policy learns to tolerate "typical" head motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Effective head-camera field of view. The head swivels to keep the ball centred,
# so "the robot can see the ball" means the ball's body-frame bearing is within
# the head's yaw reach and it is within range. Shared (single source of truth) by
# this action term, the ball perception model, and the ball-lost termination so
# all three agree on what "visible" means.
HEAD_FOV_HALF = 1.2  # rad
HEAD_MAX_RANGE = 6.0  # m


@dataclass(kw_only=True)
class HeadIKActionCfg(ActionTermCfg):
  """Configuration for IK-based head tracking with FOV return-to-zero."""

  entity_name: str = "robot"
  ball_entity: str = "ball"
  camera_name: str = "robot/d455_color"

  yaw_joint: str = "AAHead_yaw"
  pitch_joint: str = "Head_pitch"

  # Soft joint limits (stay a few degrees inside the hard limits to avoid
  # the joint-limit penalty reward term).
  yaw_limit: float = 1.5  # joint range ±1.57 rad
  pitch_min: float = -0.30  # joint min −0.35 rad
  pitch_max: float = 1.02  # joint max  1.22 rad

  # Field-of-view gate: the head tracks the ball while its body-frame bearing is
  # within ±fov_half and within max_range; otherwise it returns to (0, 0).
  # fov_half sits just inside yaw_limit so there is a clear "lost" zone, and
  # max_range matches the ball perception model's max_range.
  fov_half: float = HEAD_FOV_HALF
  max_range: float = HEAD_MAX_RANGE

  def build(self, env: ManagerBasedRlEnv) -> HeadIKAction:
    return HeadIKAction(self, env)


class HeadIKAction(ActionTerm):
  """Tracks the ball with the head camera; returns to (0, 0) when out of view."""

  cfg: HeadIKActionCfg

  def __init__(self, cfg: HeadIKActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    robot = env.scene[cfg.entity_name]

    yaw_ids, _ = robot.find_joints_by_actuator_names((cfg.yaw_joint,))
    pitch_ids, _ = robot.find_joints_by_actuator_names((cfg.pitch_joint,))
    self._head_ids = torch.tensor(
      yaw_ids + pitch_ids, device=self.device, dtype=torch.long
    )

    self._cam_id: int = env.sim.mj_model.camera(cfg.camera_name).id

    # Placeholder so raw_action is always a valid tensor.
    self._dummy = torch.zeros(self.num_envs, 0, device=self.device)

  # ------------------------------------------------------------------
  # ActionTerm interface
  # ------------------------------------------------------------------

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._dummy

  def reset(self, env_ids: torch.Tensor) -> None:
    pass  # stateless

  def process_actions(self, actions: torch.Tensor) -> None:
    pass  # nothing to process — IK is computed fresh each substep

  def apply_actions(self) -> None:
    robot = self._entity
    env = self._env

    # Camera world position (exact from MuJoCo, updated every substep).
    cam_pos = env.sim.data.cam_xpos[:, self._cam_id, :]  # (N, 3)

    ball_pos = env.scene[self.cfg.ball_entity].data.root_link_pos_w[:, :3]  # (N, 3)

    # Direction from camera to ball in world frame.
    to_ball_w = ball_pos - cam_pos
    dist = to_ball_w.norm(dim=-1)  # (N,)
    to_ball_w = to_ball_w / dist.clamp(min=1e-6).unsqueeze(-1)

    # Rotate into robot body frame (approximate neck-parent frame).
    q = robot.data.root_link_quat_w  # (N, 4)  w, x, y, z
    q_inv = torch.cat([q[:, :1], -q[:, 1:]], dim=-1)
    to_ball_b = quat_apply(q_inv, to_ball_w)  # (N, 3)

    # Head yaw — z-axis rotation, positive = turn left.
    yaw = torch.atan2(to_ball_b[:, 1], to_ball_b[:, 0])
    yaw = yaw.clamp(-self.cfg.yaw_limit, self.cfg.yaw_limit)

    # Head pitch — y-axis rotation, positive = look down.
    d_xy = (to_ball_b[:, 0] ** 2 + to_ball_b[:, 1] ** 2).sqrt().clamp(min=1e-3)
    pitch = torch.atan2(-to_ball_b[:, 2], d_xy)
    pitch = pitch.clamp(self.cfg.pitch_min, self.cfg.pitch_max)

    # Out of view -> snap the head target back to neutral (0, 0). The bearing is
    # taken before the yaw clamp so a ball beyond the head's reach counts as lost.
    bearing = torch.atan2(to_ball_b[:, 1], to_ball_b[:, 0])
    in_view = (bearing.abs() < self.cfg.fov_half) & (dist < self.cfg.max_range)
    yaw = torch.where(in_view, yaw, torch.zeros_like(yaw))
    pitch = torch.where(in_view, pitch, torch.zeros_like(pitch))

    targets = torch.stack([yaw, pitch], dim=-1)  # (N, 2)
    robot.set_joint_position_target(targets, joint_ids=self._head_ids)
