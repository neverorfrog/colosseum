from __future__ import annotations
import torch

from colosseum.deploy.config import RobotConfig

class RobotData:
    """
    The joint indexing follows the real robot,
    described in RobotCfg.joint_names
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    feedback_torque: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor

    def __init__(self, cfg: RobotConfig) -> None:
        self.cfg = cfg
        num_joints = len(self.cfg.joint_names)
        self.real2sim_joint_indexes = [cfg.joint_names.index(name) for name in cfg.sim_joint_names]
        self.sim2real_joint_indexes = [cfg.sim_joint_names.index(name) for name in cfg.joint_names]

        self.joint_pos: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.joint_vel: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.feedback_torque: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.root_lin_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_ang_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_pos_w: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_quat_w: torch.Tensor = torch.zeros(4, dtype=torch.float32)
        self.projected_gravity_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)


class BoosterRobot:
    cfg: RobotConfig
    data: RobotData
    joint_stiffness: torch.Tensor
    joint_damping: torch.Tensor
    default_joint_pos: torch.Tensor

    def __init__(self, cfg: RobotConfig) -> None:
        self.cfg = cfg
        self.data = RobotData(cfg)
        self.joint_stiffness = torch.tensor(cfg.joint_stiffness, dtype=torch.float32)
        self.joint_damping = torch.tensor(cfg.joint_damping, dtype=torch.float32)
        self.default_joint_pos = torch.tensor(cfg.default_joint_pos, dtype=torch.float32)
        self.effort_limit = torch.tensor(cfg.effort_limit, dtype=torch.float32)

    @property
    def num_joints(self) -> int:
        return len(self.cfg.joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.cfg.body_names)