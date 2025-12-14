"""Velocity task observation functions for training.

These functions extract observations from mjlab's ManagerBasedRlEnv for use
during training. They follow the VelocityObservationSpec contract.

For deployment, see tasks/velocity/deploy/<robot>/policy.py which computes
observations from sensor data.
"""

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_base import SceneEntityCfg

from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC

# Default asset config
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_ang_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Base angular velocity observation (training).

    Extracts angular velocity from entity data computed by mjlab.

    Args:
        env: Training environment.
        asset_cfg: Asset configuration (default: "robot").

    Returns:
        Batched tensor of shape (num_envs, 3) containing [wx, wy, wz].
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def base_lin_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Base linear velocity observation (training).

    Args:
        env: Training environment.
        asset_cfg: Asset configuration (default: "robot").

    Returns:
        Batched tensor of shape (num_envs, 3) containing [vx, vy, vz].
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Projected gravity observation (training).

    Extracts pre-computed projected gravity from entity data.

    Args:
        env: Training environment.
        asset_cfg: Asset configuration (default: "robot").

    Returns:
        Batched tensor of shape (num_envs, 3) containing gravity in base frame.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def joint_pos_rel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint positions relative to default (training).

    Args:
        env: Training environment.
        asset_cfg: Asset configuration (default: "robot").

    Returns:
        Batched tensor of shape (num_envs, num_joints).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def joint_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Joint velocities observation (training).

    Args:
        env: Training environment.
        asset_cfg: Asset configuration (default: "robot").

    Returns:
        Batched tensor of shape (num_envs, num_joints).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_vel


__all__ = [
    "base_ang_vel",
    "base_lin_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel",
]
