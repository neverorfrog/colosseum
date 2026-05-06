"""Manage the model registry.

Usage:
  pixi run model-registry list <task>                  # List registered models
  pixi run model-registry set-default <task> <name>    # Change default model
  pixi run model-registry remove <task> <name>         # Remove from registry only
  pixi run model-registry remove <task> <name> --delete-files  # Also delete .onnx/.pt from disk
"""

from __future__ import annotations

import argparse
import sys

from colosseum.utils.model_registry import ModelRegistry


def cmd_list(task: str) -> None:
  models = ModelRegistry.list_models(task)
  if not models:
    print(f"No models registered for task '{task}'")
    return

  print(f"\n  Task: {task}")
  for m in models:
    marker = " *" if m.get("default") else "  "
    print(f" {marker}  {m['name']:<12}  step={m['step']:>12,}  run={m['run']}")


def cmd_set_default(task: str, name: str) -> None:
  ModelRegistry.set_default(task, name)
  print(f"Default for '{task}' set to '{name}'")


def cmd_remove(task: str, name: str, delete_files: bool = False) -> None:
  ModelRegistry.remove(task, name, delete_files=delete_files)
  action = "deleted" if delete_files else "removed from registry"
  print(f"'{name}' {action} for task '{task}'")


def main() -> None:
  parser = argparse.ArgumentParser(description="Manage the model registry")
  sub = parser.add_subparsers(dest="command", required=True)

  p_list = sub.add_parser("list", help="List registered models for a task")
  p_list.add_argument("task", help="Task name")

  p_default = sub.add_parser("set-default", help="Set the default model for a task")
  p_default.add_argument("task", help="Task name")
  p_default.add_argument("name", help="Model name")

  p_remove = sub.add_parser("remove", help="Remove a model from the registry")
  p_remove.add_argument("task", help="Task name")
  p_remove.add_argument("name", help="Model name")
  p_remove.add_argument(
    "--delete-files",
    action="store_true",
    help="Also delete .onnx/.pt files from disk",
  )

  args = parser.parse_args()

  try:
    if args.command == "list":
      cmd_list(args.task)
    elif args.command == "set-default":
      cmd_set_default(args.task, args.name)
    elif args.command == "remove":
      cmd_remove(args.task, args.name, delete_files=getattr(args, "delete_files", False))
  except KeyError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
