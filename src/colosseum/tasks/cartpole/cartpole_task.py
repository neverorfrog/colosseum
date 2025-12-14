# ====== Actions and Observations ======
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.manager_term_config import ActionTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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

from colosseum.tasks.cartpole.mdp_functions import joint_pos, joint_vel

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

from colosseum.tasks.cartpole.mdp_functions import effort_cost, upright_reward

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

from colosseum.tasks.cartpole.mdp_functions import pole_fallen

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
