from typing import Callable, List, Optional, TypeVar
from dataclasses import field, MISSING
import torch
from colosseum.deploy.core.utils.isaaclab.configclass import configclass


@configclass
class PrepareStateCfg:
    stiffness: List[float] = MISSING  # type: ignore
    damping: List[float] = MISSING  # type: ignore
    joint_pos: List[float] = MISSING  # type: ignore


@configclass
class MujocoControllerCfg:
    init_pos: List[float] = [0.0, 0.0, 0.6]
    init_quat: List[float] = [1.0, 0.0, 0.0, 0.0]
    decimation: int = 10
    # physics_dt will automatically be set by ControllerCfg
    physics_dt: float = None  # type: ignore
    save_states: bool = False


@configclass
class BoosterRobotControllerCfg:
    low_state_dt: float = 0.002
    metrics_max_events: int = 2000


@configclass
class RobotCfg:
    name: str = MISSING  # type: ignore

    joint_names: list[str] = MISSING  # type: ignore
    body_names: list[str] = MISSING  # type: ignore

    sim_joint_names: list[str] = MISSING  # type: ignore
    sim_body_names: list[str] = MISSING  # type: ignore

    joint_stiffness: List[float] = MISSING  # type: ignore
    joint_damping: List[float] = MISSING  # type: ignore
    
    default_joint_pos: List[float] = MISSING  # type: ignore
    effort_limit: List[float] = MISSING  # type: ignore

    parallel_joint_indices: List[int] = MISSING  # type: ignore

    mjcf_path: str = MISSING  # type: ignore

    prepare_state: PrepareStateCfg = MISSING  # type: ignore

    def __post_init__(self):
        assert (
            len(self.joint_names)
            == len(self.joint_stiffness)
            == len(self.joint_damping)
            == len(self.default_joint_pos)
            == len(self.effort_limit)
        )


@configclass
class VelocityCommandCfg:
    vx_max: float = 1.0
    vy_max: float = 1.0
    vyaw_max: float = 1.0


@configclass
class PolicyCfg:
    constructor: Callable = MISSING  # type: ignore


@configclass
class EvaluatorCfg:
    constructor: Callable = MISSING  # type: ignore
    # Rendering
    render: bool = True


@configclass
class ControllerCfg:
    """Controller configuration class.
    """

    policy_dt: float = 0.02
    robot: RobotCfg = MISSING  # type: ignore
    vel_command: Optional[VelocityCommandCfg] = None
    policy: PolicyCfg = MISSING  # type: ignore
    mujoco: MujocoControllerCfg = MujocoControllerCfg()
    booster: BoosterRobotControllerCfg = BoosterRobotControllerCfg()
    evaluator: Optional[EvaluatorCfg] = None

    def __post_init__(self):
        self.mujoco.physics_dt = self.policy_dt / self.mujoco.decimation
