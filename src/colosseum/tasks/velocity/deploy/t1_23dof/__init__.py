"""T1 23-DOF velocity tracking deployment.

This module contains everything needed to deploy velocity tracking on T1 23-DOF:
- robot_cfg.py: Robot configuration specific to this task
- policy.py: Policy implementation
- config.py: Deployment configuration

Usage:
    >>> from colosseum.tasks.velocity.deploy.t1_23dof import T1_23DOF_VELOCITY_DEPLOY_CFG
    >>> from colosseum.deploy.core.controllers import MujocoController
    >>> controller = MujocoController(T1_23DOF_VELOCITY_DEPLOY_CFG)
    >>> controller.run()
"""

from colosseum.tasks.velocity.deploy.t1_23dof.config import (
    T1_23DOF_VELOCITY_DEPLOY_CFG,
    create_deployment_cfg,
)
from colosseum.tasks.velocity.deploy.t1_23dof.robot_cfg import T1_23DOF_VELOCITY_ROBOT_CFG
from colosseum.tasks.velocity.deploy.t1_23dof.policy import T1VelocityPolicy

__all__ = [
    "T1_23DOF_VELOCITY_DEPLOY_CFG",
    "create_deployment_cfg",
    "T1_23DOF_VELOCITY_ROBOT_CFG",
    "T1VelocityPolicy",
]
