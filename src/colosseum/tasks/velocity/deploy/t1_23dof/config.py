"""Deployment configuration for T1 23-DOF velocity tracking.

This file defines the complete deployment bundle for running velocity tracking
on the T1 23-DOF robot.
"""

from colosseum.deploy.core.controllers import (
    ControllerCfg,
    VelocityCommandCfg,
    MujocoControllerCfg,
)
from colosseum.tasks.velocity.deploy.t1_23dof.robot_cfg import T1_23DOF_VELOCITY_ROBOT_CFG
from colosseum.tasks.velocity.deploy.t1_23dof.policy import T1VelocityPolicy


def create_deployment_cfg(
    checkpoint_path: str = "models/velocity_v1.pt",
    vx_max: float = 1.0,
    vy_max: float = 0.5,
    vyaw_max: float = 1.0,
    policy_dt: float = 0.02,  # 50Hz
    decimation: int = 10,     # 500Hz physics / 50Hz policy
) -> ControllerCfg:
    """Create deployment configuration.

    Args:
        checkpoint_path: Path to trained model (relative to this file or absolute).
        vx_max: Max forward velocity (m/s).
        vy_max: Max lateral velocity (m/s).
        vyaw_max: Max yaw rate (rad/s).
        policy_dt: Policy frequency (seconds).
        decimation: Physics steps per policy step.

    Returns:
        Complete controller configuration.
    """
    # Create policy config (custom, not using PolicyCfg)
    class T1VelocityPolicyCfg:
        def __init__(self, checkpoint_path: str):
            self.checkpoint_path = checkpoint_path

        @property
        def constructor(self):
            def _constructor(cfg, controller):
                return T1VelocityPolicy(self.checkpoint_path, controller)
            return _constructor

    return ControllerCfg(
        policy_dt=policy_dt,

        # Robot configuration (T1 23-DOF specific to velocity task)
        robot=T1_23DOF_VELOCITY_ROBOT_CFG,

        # Velocity command configuration
        vel_command=VelocityCommandCfg(
            vx_max=vx_max,
            vy_max=vy_max,
            vyaw_max=vyaw_max,
        ),

        # Policy configuration
        policy=T1VelocityPolicyCfg(checkpoint_path=checkpoint_path),

        # MuJoCo simulation configuration
        mujoco=MujocoControllerCfg(
            init_pos=[0.0, 0.0, 0.6],
            init_quat=[1.0, 0.0, 0.0, 0.0],
            decimation=decimation,
        ),
    )


# Default configuration
T1_23DOF_VELOCITY_DEPLOY_CFG = create_deployment_cfg()
