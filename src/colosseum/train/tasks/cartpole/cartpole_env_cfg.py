# ====== Scene Definition ======
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainImporterCfg

from colosseum.robots.cartpole.cartpole_constants import CARTPOLE_ROBOT_CFG

scene_config = SceneCfg(
    terrain=TerrainImporterCfg(
        terrain_type="plane",
    ),
    num_envs=512,
    extent=1.0,
    entities={"robot": CARTPOLE_ROBOT_CFG},
)

# ====== Viewer Configuration ======
from mjlab.viewer import ViewerConfig

viewer_config = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="pole",
    distance=3.0,
    elevation=10.0,
    azimuth=90.0,
)


# ====== Simulation Configuration ======
from mjlab.sim import MujocoCfg, SimulationCfg

simulation_config = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.02,
        iterations=1,
    ),
)


# ====== Actions and Observations ======
from mjlab.envs.mdp.actions import JointEffortActionCfg, JointVelocityActionCfg
from mjlab.managers.manager_term_config import ActionTermCfg

actions: dict[str, ActionTermCfg] = {
    "joint_position": JointEffortActionCfg(
        asset_name="robot",
        actuator_names=("slide",),
        scale=20.0,
    ),
}

from mjlab.managers.manager_term_config import (
    ObservationGroupCfg,
    ObservationTermCfg,
)

from colosseum.train.tasks.cartpole.cartpole_mdp import joint_pos, joint_vel

policy_terms: dict[str, ObservationTermCfg] = {
    "cart_pos": ObservationTermCfg(
        func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names="slide")}
    ),
    "pole_angle": ObservationTermCfg(
        func=joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names="hinge")}
    ),
    "cart_vel": ObservationTermCfg(
        func=joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="slide")}
    ),
    "pole_ang_vel": ObservationTermCfg(
        func=joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="hinge")}
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


from mjlab.envs.mdp.terminations import time_out

# ======== Rewards ========
from mjlab.managers.manager_term_config import RewardTermCfg

from colosseum.train.tasks.cartpole.cartpole_mdp import (
    effort_cost,
    upright_reward,
)

rewards: dict[str, RewardTermCfg] = {
    "upright": RewardTermCfg(
        func=upright_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="hinge"),
            "dummy_param": 0.5,
        },
    ),
    "effort": RewardTermCfg(
        func=effort_cost, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot")}
    ),
}

# ====== Terminations =======
from mjlab.managers.manager_term_config import TerminationTermCfg

from colosseum.train.tasks.cartpole.cartpole_mdp import pole_fallen

terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "fallen_pole": TerminationTermCfg(
        func=pole_fallen,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="hinge")},
    ),
}


from mjlab.envs.mdp import randomize_field

# ====== Events ============
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.managers.manager_term_config import EventTermCfg

events: dict[str, EventTermCfg] = {
    "reset_joints": EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    ),
    "base_com": EventTermCfg(  # important
        func=randomize_field,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "abs",
            "field": "body_mass",
            "ranges": (0.8, 1.2),
        },
    ),
}

# ==== Env Config =======
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg

env_config = ManagerBasedRlEnvCfg(
    scene=scene_config,
    observations=observations,
    actions=actions,
    rewards=rewards,
    events=events,
    terminations=terminations,
    sim=simulation_config,
    viewer=viewer_config,
    decimation=1,
    episode_length_s=10.0,
)


# === RL Agent =======
from mjlab.rl import RslRlOnPolicyRunnerCfg

rl_config = RslRlOnPolicyRunnerCfg(max_iterations=500, wandb_project="mjlab_cartpole")
