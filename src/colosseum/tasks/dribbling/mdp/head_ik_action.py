"""IK-based head tracking action term for the dribbling task.

Computes head yaw and pitch analytically from the ball position so the
camera stays pointed at the ball without relying on reward-shaped learning.
Consumes no policy output (action_dim=0) and overrides the head joint
targets on every decimation substep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HeadIKActionCfg(ActionTermCfg):
  """Configuration for IK-based head tracking.

  The term reads the ball world position each substep and writes
  analytically computed yaw/pitch targets for the two neck joints.
  No policy output is consumed (action_dim=0).
  """

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

  # Periodic scan behaviour: look straight ahead and slightly up for
  # obstacle awareness. Durations are in seconds (converted to substeps
  # in __init__ using the physics timestep).
  scan_yaw: float = 0.0  # look straight ahead (rad)
  scan_pitch: float = 0.1  # look slightly up (rad, negative = up)
  track_duration: float = 2.0  # seconds tracking the ball
  scan_duration: float = 0.3  # seconds looking up

  def build(self, env: ManagerBasedRlEnv) -> HeadIKAction:
    return HeadIKAction(self, env)


class HeadIKAction(ActionTerm):
  """Analytically tracks the ball with the head camera.

  On every apply_actions() call (inside the decimation loop) this term:
    1. Reads the ball world position and the camera world position.
    2. Computes the camera→ball unit vector, rotated into the robot body frame.
    3. Extracts yaw (z-axis) and pitch (y-axis, positive = look down).
    4. Clamps to soft joint limits and writes the targets.

  Using the camera position as the IK pivot (rather than the robot base)
  gives a more accurate pointing direction since the camera is ~1.2 m above
  and forward of the base.  The robot's base quaternion approximates the
  neck-parent frame orientation — accurate enough for a soft PD controller.
  """

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

    # Per-env phase counter (in substeps). Each env independently cycles
    # through [0, track_steps + scan_steps). Scan fires in the last
    # scan_steps of the cycle. Phase is randomised at init so envs are
    # desynchronised from the start.
    dt = env.sim.mj_model.opt.timestep
    self._track_steps = max(1, round(cfg.track_duration / dt))
    self._scan_steps = max(1, round(cfg.scan_duration / dt))
    total = self._track_steps + self._scan_steps
    self._counter = torch.randint(0, total, (self.num_envs,), device=self.device)

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
    total = self._track_steps + self._scan_steps
    self._counter[env_ids] = torch.randint(
      0, total, (len(env_ids),), device=self.device
    )

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
    to_ball_w = to_ball_w / to_ball_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)

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

    targets = torch.stack([yaw, pitch], dim=-1)  # (N, 2)

    # Advance phase counter and override targets for envs in scan phase.
    total = self._track_steps + self._scan_steps
    self._counter = (self._counter + 1) % total
    in_scan = self._counter >= self._track_steps  # (N,) bool
    targets[in_scan, 0] = self.cfg.scan_yaw
    targets[in_scan, 1] = self.cfg.scan_pitch

    robot.set_joint_position_target(targets, joint_ids=self._head_ids)
