#!/usr/bin/env python3
"""Deployment script for Colosseum tasks.

This script supports:
- MuJoCo simulation (--mujoco)
- Real robot deployment (default, requires Booster SDK)
- Webots simulation (--webots, requires Webots)

Usage:
    # List available tasks
    python scripts/deploy.py --list

    # Run in MuJoCo simulation
    python scripts/deploy.py --task t1_23dof_velocity --mujoco

    # Run on real robot
    python scripts/deploy.py --task t1_23dof_velocity --net 192.168.123.161

Based on booster_deploy/scripts/deploy.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

parser = argparse.ArgumentParser(
    description="Deploy Colosseum tasks on simulation or real robots"
)

# Task selection (mutually exclusive with --list)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--task", type=str, help="Task name (e.g., 't1_23dof_velocity')")
group.add_argument(
    "-l",
    "--list",
    action="store_true",
    dest="list_tasks",
    help="List all available tasks",
)

# Deployment target
parser.add_argument(
    "--mujoco", action="store_true", default=False, help="Deploy in MuJoCo simulation"
)
parser.add_argument(
    "--webots", action="store_true", default=False, help="Deploy in Webots simulation"
)

# Real robot options
parser.add_argument(
    "--net",
    type=str,
    default="127.0.0.1",
    help="Network interface for SDK communication (real robot)",
)

args = parser.parse_args()


def discover_tasks():
    """Discover all available task deployments.

    Scans tasks/*/deploy/*/ for deployment bundles.

    Returns:
        dict: Mapping of task names to deployment configs.
    """
    tasks_dir = project_root / "src" / "colosseum" / "tasks"
    discovered_tasks = {}

    if not tasks_dir.exists():
        return discovered_tasks

    # Scan tasks/*/deploy/*/
    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue

        deploy_dir = task_dir / "deploy"
        if not deploy_dir.exists():
            continue

        # Each subdirectory is a robot-specific deployment
        for robot_deploy_dir in deploy_dir.iterdir():
            if not robot_deploy_dir.is_dir() or robot_deploy_dir.name.startswith("_"):
                continue

            # Check if it has __init__.py with deployment config
            init_file = robot_deploy_dir / "__init__.py"
            if not init_file.exists():
                continue

            # Try to import and get default config
            task_name = f"{task_dir.name}_{robot_deploy_dir.name}"
            module_path = (
                f"colosseum.tasks.{task_dir.name}.deploy.{robot_deploy_dir.name}"
            )

            try:
                import importlib

                module = importlib.import_module(module_path)

                # Look for *_DEPLOY_CFG pattern
                for attr_name in dir(module):
                    if attr_name.endswith("_DEPLOY_CFG") and not attr_name.startswith(
                        "_"
                    ):
                        cfg = getattr(module, attr_name)
                        discovered_tasks[task_name] = {
                            "config": cfg,
                            "module": module_path,
                            "attr": attr_name,
                        }
                        break
            except Exception as e:
                print(f"Warning: Failed to import {module_path}: {e}", file=sys.stderr)

    return discovered_tasks


def main():
    # Discover available tasks
    print("Discovering tasks...")
    tasks = discover_tasks()

    if args.list_tasks:
        print("\nAvailable tasks:")
        if not tasks:
            print("  (none found)")
            print("\nNote: Task deployments should be in tasks/<task>/deploy/<robot>/")
        for task_name, info in sorted(tasks.items()):
            print(f"  {task_name:30s} : {info['module']}.{info['attr']}")
        sys.exit(0)

    # Get requested task
    if args.task not in tasks:
        print(f"Error: Unknown task '{args.task}'")
        print(f"\nAvailable tasks: {list(tasks.keys())}")
        print("\nUse --list to see all available tasks")
        sys.exit(1)

    task_cfg = tasks[args.task]["config"]
    print(f"\nLoading task: {args.task}")
    print(f"  Robot: {task_cfg.robot.name}")
    print(f"  Joints: {len(task_cfg.robot.joint_names)}")

    # Decide deployment target
    if args.mujoco:
        print("\n=== Running in MuJoCo simulation ===\n")
        from colosseum.deploy.core.backends import MujocoController

        controller = MujocoController(task_cfg)
        controller.run()

    elif args.webots:
        print("\n=== Running in Webots simulation ===\n")
        # Initialize SDK for Webots
        try:
            from booster_robotics_sdk_python import ChannelFactory

            ChannelFactory.Instance().Init(0, args.net)
        except ImportError:
            print(
                "Error: booster_robotics_sdk_python not installed.\n"
                "Please install Booster SDK for Webots simulation.\n"
                "For MuJoCo simulation, use --mujoco instead."
            )
            sys.exit(1)

        # Adjust ankle dampings for Webots
        ankles = [-8, -7, -2, -1]  # Indices of ankle joints
        for i in ankles:
            task_cfg.robot.joint_damping[i] = 0.5

        from colosseum.deploy.core.backends.booster import (
            BoosterRobotPortal,
        )

        with BoosterRobotPortal(task_cfg, use_sim_time=True) as portal:
            portal.run()

    else:
        print(f"\n=== Running on real robot (net: {args.net}) ===\n")
        # Initialize SDK for real robot
        try:
            from booster_robotics_sdk_python import ChannelFactory

            ChannelFactory.Instance().Init(0, args.net)
        except ImportError:
            print(
                "Error: booster_robotics_sdk_python not installed.\n"
                "Please install Booster SDK to use real robot deployment.\n"
                "For MuJoCo simulation, use --mujoco flag."
            )
            sys.exit(1)

        from colosseum.deploy.core.backends.booster import (
            BoosterRobotPortal,
        )

        with BoosterRobotPortal(task_cfg, use_sim_time=False) as portal:
            portal.run()


if __name__ == "__main__":
    main()
