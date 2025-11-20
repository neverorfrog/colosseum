from mjlab.scene import SceneCfg
from colosseum.robots.cartpole.cartpole_constants import get_cartpole_robot_cfg

SCENE_CFG: SceneCfg = SceneCfg(
    num_envs=512,
    extent=1.0,
    entities={"robot": get_cartpole_robot_cfg()},
)

from mjlab.viewer import ViewerConfig

VIEWER_CFG: ViewerConfig = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    asset_name="robot",
    distance=3.0,
    elevation=10.0,
)

from mjlab.sim import MujocoCfg, SimulationCfg

SIM_CFG: SimulationCfg = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.02,
        iterations=10,
    ),
)

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ActionTermCfg

actions: dict[str, ActionTermCfg] = {
    "joint_position": JointPositionActionCfg(
        asset_name="robot",
        actuator_names=(".*",),
        scale=20.0,
    ),
}

from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity
import torch

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

policy_terms: dict[str, ObservationTermCfg] = {
    "cart_pos": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 0:1]),
    "angle": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 1:2]),
    "cart_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 0:1]),
    "ang_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 1:2]),
}

def joint_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_pos[:, jnt_ids]

def joint_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids]


policy_terms: dict[str, ObservationTermCfg] = {
    "cart_pos": ObservationTermCfg(
        func=joint_pos,
        params={"asset_cfg": SceneEntityCfg("cart", joint_names="slide")}
    ),
    "pole_angle": ObservationTermCfg(
        func=joint_pos,
        params={"asset_cfg": SceneEntityCfg("pole", joint_names="hinge")}
    ),
    "cart_vel": ObservationTermCfg(
        func=joint_vel,
        params={"asset_cfg": SceneEntityCfg("cart", joint_names="slide")}
    ),
    "pole_ang_vel": ObservationTermCfg(
        func=joint_vel,
        params={"asset_cfg": SceneEntityCfg("pole", joint_names="hinge")}
    ),
}

critic_terms: dict[str, ObservationTermCfg] = {**policy_terms}

observations: dict[str, ObservationGroupCfg] = {
"policy": ObservationGroupCfg(
    terms=policy_terms,
    concatenate_terms=True,
    enable_corruption=True,
),
"critic": ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
),
}





