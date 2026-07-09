"""Privileged observation functions that read DR-randomized model parameters."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def foot_friction_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Foot friction coefficient randomized by DR (1D per env).

  Reads the current geom_friction value for the first resolved foot geom.
  When the DR event uses shared_random=True all foot geoms have the same
  value, so a single sample suffices.

  Args:
    asset_cfg: SceneEntityCfg with geom_names matching the foot geoms.

  Returns:
    (N, 1) friction coefficient tensor.
  """
  geom_ids = asset_cfg.geom_ids
  gid = geom_ids[0] if isinstance(geom_ids, (list, tuple)) else int(geom_ids)
  return env.sim.model.geom_friction[:, gid, 0:1]  # (N, 1)


def base_com_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Base body COM offset from its parent frame origin (3D per env).

  The DR event adds a random offset to body_ipos, which stores the COM
  position relative to the parent body frame in the MuJoCo model.

  Args:
    asset_cfg: SceneEntityCfg with body_names=(BASE_BODY_NAME,).

  Returns:
    (N, 3) COM offset tensor.
  """
  bid = asset_cfg.body_ids[0] if isinstance(asset_cfg.body_ids, (list, tuple)) else int(asset_cfg.body_ids)
  return env.sim.model.body_ipos[:, bid, :]  # (N, 3)


def actuator_kp_scale_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Mean PD proportional gain scale across resolved actuators (1D per env).

  Reads current gainprm and divides by the pre-DR default to recover the
  multiplicative scale applied by the pd_gains DR event.

  Args:
    asset_cfg: SceneEntityCfg resolving the actuators to monitor.

  Returns:
    (N, 1) mean kp scale tensor.
  """
  ctrl_ids = asset_cfg.actuator_ids
  if isinstance(ctrl_ids, slice):
    n = env.sim.model.actuator_gainprm.shape[1]
    ctrl_ids = list(range(*ctrl_ids.indices(n)))
  kp_cur = env.sim.model.actuator_gainprm[:, ctrl_ids, 0]       # (N, A)
  kp_def = env.sim.get_default_field("actuator_gainprm")[ctrl_ids, 0]  # (A,)
  return (kp_cur / kp_def.clamp(min=1e-6)).mean(dim=-1, keepdim=True)  # (N, 1)


def actuator_kd_scale_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Mean PD derivative gain scale across resolved actuators (1D per env).

  Reads current biasprm[2] and divides by the pre-DR default.

  Args:
    asset_cfg: SceneEntityCfg resolving the actuators to monitor.

  Returns:
    (N, 1) mean kd scale tensor.
  """
  ctrl_ids = asset_cfg.actuator_ids
  if isinstance(ctrl_ids, slice):
    n = env.sim.model.actuator_biasprm.shape[1]
    ctrl_ids = list(range(*ctrl_ids.indices(n)))
  kd_cur = env.sim.model.actuator_biasprm[:, ctrl_ids, 2].abs()       # (N, A)
  kd_def = env.sim.get_default_field("actuator_biasprm")[ctrl_ids, 2].abs()  # (A,)
  return (kd_cur / kd_def.clamp(min=1e-6)).mean(dim=-1, keepdim=True)  # (N, 1)


def link_mass_scale_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Mean body mass scale across resolved bodies (1D per env).

  Reads current body_mass and divides by the pre-DR default to recover the
  multiplicative scale applied by the pseudo_inertia DR event.

  Args:
    asset_cfg: SceneEntityCfg with body_names matching robot bodies.

  Returns:
    (N, 1) mean mass scale tensor.
  """
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice):
    n = env.sim.model.body_mass.shape[1]
    body_ids = list(range(*body_ids.indices(n)))
  mass_cur = env.sim.model.body_mass[:, body_ids]           # (N, B)
  mass_def = env.sim.get_default_field("body_mass")[body_ids]  # (B,)
  return (mass_cur / mass_def.clamp(min=1e-6)).mean(dim=-1, keepdim=True)  # (N, 1)


def joint_damping_scale_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Mean joint damping scale across resolved joints (1D per env).

  Reads current dof_damping and divides by the pre-DR default to recover the
  multiplicative scale applied by the joint_damping DR event.

  Args:
    asset_cfg: SceneEntityCfg with joint_names matching robot joints.

  Returns:
    (N, 1) mean damping scale tensor.
  """
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    n = env.sim.model.dof_damping.shape[1]
    joint_ids = list(range(*joint_ids.indices(n)))
  damp_cur = env.sim.model.dof_damping[:, joint_ids]           # (N, J)
  damp_def = env.sim.get_default_field("dof_damping")[joint_ids]  # (J,)
  return (damp_cur / damp_def.clamp(min=1e-6)).mean(dim=-1, keepdim=True)  # (N, 1)


def effort_limit_scale_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Mean effort limit scale across resolved actuators (1D per env).

  Reads the positive forcerange limit and divides by the pre-DR default to
  recover the scale applied by the effort_limits DR event.

  Args:
    asset_cfg: SceneEntityCfg resolving all actuators.

  Returns:
    (N, 1) mean effort limit scale tensor.
  """
  ctrl_ids = asset_cfg.actuator_ids
  if isinstance(ctrl_ids, slice):
    n = env.sim.model.actuator_forcerange.shape[1]
    ctrl_ids = list(range(*ctrl_ids.indices(n)))
  force_cur = env.sim.model.actuator_forcerange[:, ctrl_ids, 1]           # (N, A) positive limit
  force_def = env.sim.get_default_field("actuator_forcerange")[ctrl_ids, 1]  # (A,)
  return (force_cur / force_def.clamp(min=1e-6)).mean(dim=-1, keepdim=True)  # (N, 1)
