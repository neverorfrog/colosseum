from typing import Callable, List, Optional, TypeVar
from dataclasses import field
import torch
from colosseum.deploy.core.utils.isaaclab.configclass import configclass


@configclass
class PrepareStateCfg:
    stiffness: List[float] = field(default_factory=list)
    damping: List[float] = field(default_factory=list)
    joint_pos: List[float] = field(default_factory=list)


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
    name: str = field(default_factory=str)

    joint_names: list[str] = field(default_factory=list)
    body_names: list[str] = field(default_factory=list)

    sim_joint_names: list[str] = field(default_factory=list)
    sim_body_names: list[str] = field(default_factory=list)

    joint_stiffness: List[float] = field(default_factory=list)
    joint_damping: List[float] = field(default_factory=list)
    
    default_joint_pos: List[float] = field(default_factory=list)
    effort_limit: List[float] = field(default_factory=list)

    parallel_joint_indices: List[int] = field(default_factory=list)

    mjcf_path: str = field(default_factory=str)

    prepare_state: PrepareStateCfg = field(default_factory=PrepareStateCfg)

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
    constructor: Callable


@configclass
class EvaluatorCfg:
    constructor: Callable
    # Rendering
    render: bool = True


@configclass
class ControllerCfg:
    """Controller configuration class.
    """

    policy_dt: float = 0.02
    robot: RobotCfg = field(default_factory=RobotCfg)
    vel_command: Optional[VelocityCommandCfg] = None
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    mujoco: MujocoControllerCfg = MujocoControllerCfg()
    booster: BoosterRobotControllerCfg = BoosterRobotControllerCfg()
    evaluator: Optional[EvaluatorCfg] = None

    def __post_init__(self):
        self.mujoco.physics_dt = self.policy_dt / self.mujoco.decimation
