import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def randomize_actuator_kp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Scale position actuator gains (kp = actuator_gainprm[:, 0]) per environment.

  This is the Kp of the PD controller. mjlab's dr module has no equivalent
  because gainprm is not a standard DR target — it requires writing directly
  to the batched model field.
  """
  asset = env.scene[asset_cfg.name]
  actuator_ids = asset_cfg.actuator_ids  # verify attr name in your mjlab version

  lo, hi = scale_range
  n = len(env_ids)
  n_act = len(actuator_ids)

  # Default kp values (gainprm[:, 0] for position actuators)
  default_kp = asset.data.default_actuator_gainprm[env_ids][:, actuator_ids, 0]

  scale = torch.rand(n, n_act, device=env.device) * (hi - lo) + lo
  new_kp = default_kp * scale

  # Write to the batched model — exact API depends on your mjlab version
  env.sim.model.actuator_gainprm[env_ids[:, None], actuator_ids[None, :], 0] = new_kp
