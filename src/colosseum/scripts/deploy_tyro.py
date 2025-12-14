#!/usr/bin/env python3
"""Colosseum deployment CLI using Tyro.

This script provides a CLI for deploying trained policies on robots or in simulation.

Usage:
    pixi run deploy -b sim -t t1-velocity
    pixi run deploy -b robot -t t1-velocity --network-interface eth0

TODO: Add Webots support
TODO: Check with deploy.py script for consistency
"""

import tyro
from typing import Literal, Optional, Annotated
from dataclasses import replace

from colosseum.deploy.utils.export import resolve_policy_artifact
from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks

# Automatically discover and register all tasks
auto_register_tasks()

def main(
    task: Annotated[str, tyro.conf.arg(aliases=["-t"])] = "t1-velocity-rough",
    backend: Annotated[Literal["mujoco", "robot", "webots"], tyro.conf.arg(aliases=["-b"])] = "mujoco",
    checkpoint_path: Optional[str] = None,
    network_interface: str = "lo",
    domain_id: int = 0,
) -> None:
    """Deploy trained policy on robot or simulation.

    Args:
        task: Task name from registry (e.g., 't1-velocity')
        backend: Deployment backend ('sim' for MuJoCo, 'robot' for real hardware)
        checkpoint_path: Override checkpoint path (default: from task config)
        network_interface: Network interface for real robot (default: 'lo')
        domain_id: ROS2 domain ID for real robot (default: 0)
    """
    # Get task configuration from registry
    try:
        config = TASK_REGISTRY.get_config(task)
    except ValueError as e:
        print(f"[Deploy] Error: {e}")
        print(f"[Deploy] Available tasks: {list(TASK_REGISTRY.list_tasks().keys())}")
        return

    policy_cfg = config.policy

    # Override checkpoint path if provided
    if checkpoint_path:
        policy_cfg = replace(policy_cfg, checkpoint_path=checkpoint_path)

    artifact_path = resolve_policy_artifact(policy_cfg)
    policy_cfg = replace(policy_cfg, checkpoint_path=str(artifact_path))
    config = replace(config, policy=policy_cfg)

    print(f"[Deploy] Task: {task}")
    print(f"[Deploy] Robot: {config.robot.name}")
    print(f"[Deploy] Policy: {config.policy.task_name}")
    print(f"[Deploy] Checkpoint: {config.policy.checkpoint_path}")
    print(f"[Deploy] Backend: {backend}")

    if backend == "mujoco":
        # MuJoCo simulation deployment
        from colosseum.deploy.backends.mujoco import MujocoController

        print(f"[Deploy] Starting MuJoCo simulation...")
        print(f"[Deploy] Initial position: {config.mujoco.init_pos}")
        print(f"[Deploy] Decimation: {config.mujoco.decimation} (physics_dt={config.physics_dt:.4f}s)")

        controller = MujocoController(config)
        controller.run()

    else:
        # Real robot deployment
        from colosseum.deploy.backends.booster import BoosterRobotPortal

        print(f"[Deploy] Starting real robot deployment...")
        print(f"[Deploy] Network interface: {network_interface}")
        print(f"[Deploy] ROS2 domain ID: {domain_id}")

        with BoosterRobotPortal(config) as portal:
            portal.run()


if __name__ == "__main__":
    tyro.cli(main)
