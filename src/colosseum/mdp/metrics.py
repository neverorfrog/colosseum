"""Layer 1: Universal metric functions (robot-agnostic).

Diagnostic per-step scalars logged by mjlab's ``MetricsManager`` under the
``Episode_Metrics/<name>`` prefix (per-episode mean, no weight, no dt scaling,
no gradient). Unlike reward terms — whose logged values are weighted and shaped
— these are reported in physical units, so a gait can be read straight off the
wandb plots.

Each function takes ``env`` first and returns a ``[num_envs]`` tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# =========================
# Posture / balance
# =========================
def root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Base height above the terrain floor (m)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def base_tilt(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Uprightness: norm of projected gravity in the base XY plane (0 = upright)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)


# =========================
# Velocity tracking (physical units)
# =========================
def forward_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Actual forward (base-x) linear velocity (m/s)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, 0]


def cmd_lin_vel_x(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Commanded base-x linear velocity (m/s)."""
  return env.command_manager.get_command(command_name)[:, 0]


def cmd_lin_vel_y(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Commanded base-y linear velocity (m/s)."""
  return env.command_manager.get_command(command_name)[:, 1]


def cmd_ang_vel_yaw(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Commanded yaw angular velocity (rad/s)."""
  return env.command_manager.get_command(command_name)[:, 2]


def lin_vel_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Magnitude of the planar linear-velocity tracking error (m/s)."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  return torch.norm(cmd[:, :2] - asset.data.root_link_lin_vel_b[:, :2], dim=-1)


def ang_vel_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute yaw-rate tracking error (rad/s)."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  return torch.abs(cmd[:, 2] - asset.data.root_link_ang_vel_b[:, 2])


# =========================
# Per-joint-group effort (register one term per group via asset_cfg)
# =========================
def joint_vibration(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Mean absolute joint acceleration over the group (rad/s²).

  A proxy for vibration / chatter — high-frequency oscillation shows up as
  large accelerations.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_acc[:, asset_cfg.joint_ids].abs().mean(dim=-1)


def joint_torque(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Mean absolute actuator torque over the group (N·m).

  Uses ``qfrc_actuator`` (actuator force mapped into joint space): for the T1
  motors this is the commanded torque times the gear ratio.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.qfrc_actuator[:, asset_cfg.joint_ids].abs().mean(dim=-1)


class joint_jerk:
  """Mean absolute joint jerk over the group (rad/s³).

  Finite difference of joint acceleration between consecutive steps,
  ``d(joint_acc)/dt``. Captures vibration that mean ``|acceleration|`` misses:
  a large but smooth acceleration scores low jerk, while buzzing / chatter
  (acceleration rapidly reversing sign) scores high.

  Stateful: caches the previous step's acceleration. The first step of each
  episode reports 0 (no previous sample), avoiding a reset spike.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._prev_acc: torch.Tensor | None = None
    self._valid: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    acc = asset.data.joint_acc[:, asset_cfg.joint_ids]  # (N, J)
    if self._prev_acc is None or self._prev_acc.shape != acc.shape:
      self._prev_acc = torch.zeros_like(acc)
      self._valid = torch.zeros(acc.shape[0], dtype=torch.bool, device=acc.device)
    assert self._valid is not None
    jerk = (acc - self._prev_acc).abs().mean(dim=-1) / env.step_dt  # (N,)
    out = jerk * self._valid.float()  # zero on the first post-reset step
    self._prev_acc.copy_(acc)
    self._valid.fill_(True)
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._prev_acc is None:
      return
    assert self._valid is not None
    if env_ids is None:
      env_ids = slice(None)
    self._prev_acc[env_ids] = 0.0
    self._valid[env_ids] = False


# =========================
# Gait / foot kinematics
# =========================
def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Mean of the last completed swing (air) duration across feet (s).

  A cadence proxy: longer air time = slower, longer strides. Requires a
  contact sensor with ``track_air_time=True``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.last_air_time is not None
  return sensor.data.last_air_time.mean(dim=-1)


def feet_clearance(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
) -> torch.Tensor:
  """Mean terrain-relative foot height across feet (m) — swing clearance."""
  sensor = env.scene[height_sensor_name]
  return sensor.data.heights.mean(dim=-1)


def contact_force_peak(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Peak contact-force magnitude over the control step (N).

  Reads the sensor's per-substep ``force_history`` ([B, P, H, 3]) and takes the
  max magnitude across primaries and substeps. A kick is a brief impulse that can
  resolve mid-decimation-loop and read ~0 in the instantaneous ``force`` (the
  last substep only); the history peak catches it. This is the quantity a robust
  kick gate should threshold on, so logging it shows the real strike forces and
  what ``min_contact_force`` needs to be. Requires ``history_length > 0`` on the
  sensor (foot_ball_contact has 4 = decimation).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  fh = sensor.data.force_history
  assert fh is not None, f"sensor '{sensor_name}' needs history_length>0"
  return fh.norm(dim=-1).amax(dim=(1, 2))


def feet_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Mean foot-ground contact-force magnitude across feet (N) — landing impact."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  return sensor.data.force.norm(dim=-1).mean(dim=-1)


def feet_lateral_distance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Body-frame lateral (y) distance between the two feet (m) — stance width.

  The two foot world positions are differenced and rotated into the base frame,
  so the value is the side-to-side foot separation regardless of heading.
  ``asset_cfg.site_ids`` must resolve to exactly the two foot sites.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :3]  # (N, 2, 3)
  rel_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]  # (N, 3)
  quat_w = asset.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  rel_b = quat_apply(quat_conj, rel_w)
  return rel_b[:, 1].abs()


# =========================
# Joint limits
# =========================
def joint_pos_limits_violation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Fraction of joints in the group exceeding soft joint position limits.

  Returns a [num_envs] tensor with values in [0, 1].
  """
  asset: Entity = env.scene[asset_cfg.name]
  soft = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]  # (N, J, 2)
  pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, J)
  lower_violation = (pos < soft[:, :, 0]).float()
  upper_violation = (pos > soft[:, :, 1]).float()
  return (lower_violation + upper_violation).mean(dim=-1)
