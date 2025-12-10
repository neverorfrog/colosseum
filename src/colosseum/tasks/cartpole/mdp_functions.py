import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


# ====== Observation Functions ======
def joint_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    return asset.data.joint_pos[:, jnt_ids]


def joint_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    return asset.data.joint_vel[:, jnt_ids]


# ====== Reward Functions ======
def upright_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dummy_param: float = 1.0,
) -> torch.Tensor:
    del dummy_param
    asset: Entity = env.scene[asset_cfg.name]
    pole_angle = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    reward = torch.cos(pole_angle)  # 1 when upright, -1 when inverted
    return reward


def effort_cost(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    forces = asset.data.actuator_force
    effort_squared = torch.sum(torch.square(forces), dim=1)
    return effort_squared


# ====== Termination Functions ======
def pole_fallen(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    angle_threshold: float = torch.pi / 4,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    pole_angle = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    return torch.abs(pole_angle) > angle_threshold
