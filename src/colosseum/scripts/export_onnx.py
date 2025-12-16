"""CLI utility to convert a deployment policy to ONNX using task presets."""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, Optional

import tyro

from colosseum.utils.train.export import export_policy
from colosseum.deploy.core.registry import TASK_REGISTRY, auto_register_tasks


def main(
  task: Annotated[str, tyro.conf.arg(aliases=["-t"])] = "t1-velocity",
  checkpoint_path: Optional[str] = None,
  export_checkpoint_path: Optional[str] = None,
  output_dir: Optional[str] = None,
  filename: Optional[str] = None,
) -> None:
  """Export a policy preset to ONNX.

  Args:
    task: Task identifier registered in ``TASK_REGISTRY``.
    checkpoint_path: Override the ONNX destination path.
    export_checkpoint_path: Override the source RSL-RL checkpoint.
    output_dir: Override the directory where the ONNX file is stored.
    filename: Override the ONNX filename (defaults to ``policy.onnx``).
  """

  auto_register_tasks()
  config = TASK_REGISTRY.get_config(task)
  policy_cfg = config.policy

  if checkpoint_path:
    policy_cfg = replace(policy_cfg, checkpoint_path=checkpoint_path)
  if export_checkpoint_path:
    policy_cfg = replace(policy_cfg, export_checkpoint_path=export_checkpoint_path)
  if output_dir:
    policy_cfg = replace(policy_cfg, export_output_dir=output_dir)
  if filename:
    policy_cfg = replace(policy_cfg, export_filename=filename)

  artifact = export_policy(policy_cfg)
  print(f"[Export] Task: {task}")
  print(f"[Export] Source checkpoint: {policy_cfg.export_checkpoint_path}")
  print(f"[Export] Destination: {artifact}")


if __name__ == "__main__":
  tyro.cli(main)