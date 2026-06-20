#!/usr/bin/env python3
"""Distill a deploy ``gains.yaml`` from a trained run's ``config.yaml``.

The per-joint PD gains, armature, friction, effort limit, default pose and action
scale a policy was trained with are part of its deploy contract: a checkpoint is
only valid with the gains it was trained on. They already travel with every model
(``config.yaml`` next to the ONNX); this script flattens the relevant fields into a
compact, name-keyed ``gains.yaml`` that the C++ deploy repos (arena,
spqrbooster2026) load at runtime instead of hardcoding.

  pixi run export-gains models/t1-velocity-manu/jun19_1
  pixi run export-gains models/t1-velocity-manu/*        # backfill every version

Output (one field per line, joint-name keyed -> robust to joint-order drift):

  Left_Hip_Pitch.kp: 33.09
  Left_Hip_Pitch.kd: 3.950
  Left_Hip_Pitch.armature: 0.052391
  Left_Hip_Pitch.frictionloss: 0.03
  Left_Hip_Pitch.effort_limit: 90.0
  Left_Hip_Pitch.default_pos: -0.24
  Left_Hip_Pitch.action_scale: 0.6799
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tyro
import yaml
from loguru import logger

# frictionloss is left null in config.yaml (the value lives in the robot XML); the
# deploy repos use this constant, so emit it when config does not pin a value.
DEFAULT_FRICTIONLOSS = 0.03

# Fields, in emit order. Documented here so both consumers agree on the schema.
FIELDS = ("kp", "kd", "armature", "frictionloss", "effort_limit", "default_pos", "action_scale")


def _find_robot_entity(node: object) -> dict | None:
  """Return the entity node holding ``articulation.actuators``.

  The robot entity carries both its ``articulation.actuators`` and its
  ``init_state.joint_pos`` as siblings, so finding this node ties the actuator
  gains to the correct default pose (other ``joint_pos`` blocks in the config —
  randomization ranges, env-level defaults — must not be used).
  Path-robust: we search by key rather than hardcoding the deep mjlab path.
  """
  if isinstance(node, dict):
    art = node.get("articulation")
    if isinstance(art, dict):
      acts = art.get("actuators")
      if isinstance(acts, list) and acts and isinstance(acts[0], dict) and "target_names_expr" in acts[0]:
        return node
    for v in node.values():
      found = _find_robot_entity(v)
      if found is not None:
        return found
  elif isinstance(node, list):
    for v in node:
      found = _find_robot_entity(v)
      if found is not None:
        return found
  return None


def _is_pattern(name: str) -> bool:
  """True if a name is a regex pattern rather than a concrete joint name."""
  return any(c in name for c in ".*+?[]()^$|\\")


def _find_action_scale(node: object) -> dict | float | None:
  """Return the joint-position action term's ``scale`` (per-joint dict or scalar).

  Identified by the ``use_default_offset`` key (unique to JointPositionAction).
  This is the scale the policy was trained with — the source of truth, not a
  derived formula (it is 0.25 flat for the locomotion set, per-joint for manu).
  """
  if isinstance(node, dict):
    if "use_default_offset" in node and "scale" in node:
      return node["scale"]
    for v in node.values():
      found = _find_action_scale(v)
      if found is not None:
        return found
  elif isinstance(node, list):
    for v in node:
      found = _find_action_scale(v)
      if found is not None:
        return found
  return None


def _gains_from_config(config_path: Path) -> dict[str, dict[str, float]]:
  cfg = yaml.safe_load(config_path.read_text())

  robot = _find_robot_entity(cfg)
  if robot is None:
    raise ValueError(f"{config_path}: no robot entity with actuators found")

  actuators = robot["articulation"]["actuators"]
  joint_pos = robot.get("init_state", {}).get("joint_pos", {})

  # Action scale the policy trained with (per-joint dict or a single scalar).
  action_scale = _find_action_scale(cfg)
  if action_scale is None:
    raise ValueError(f"{config_path}: no joint-position action term found")

  # Concrete joint set (and default pose) comes from the entity's init_state.
  # Actuators may target joints individually (manufacturer set) or by regex group
  # (locomotion set, e.g. ".*Hip_Pitch"); expand patterns against these names.
  joint_names = [j for j in joint_pos if not _is_pattern(j)]
  if not joint_names:
    raise ValueError(f"{config_path}: no concrete joints in init_state.joint_pos")

  joint_to_act: dict[str, dict] = {}
  for act in actuators:
    for pattern in act["target_names_expr"]:
      for joint in joint_names:
        if re.fullmatch(pattern, joint):
          joint_to_act[joint] = act

  gains: dict[str, dict[str, float]] = {}
  for joint in joint_names:
    act = joint_to_act.get(joint)
    if act is None:
      continue  # joint has no actuator (held passive) — nothing to deploy
    friction = act.get("frictionloss")
    # action_scale: per-joint dict entry, or the scalar if scale is a single value.
    scale = action_scale.get(joint, 0.0) if isinstance(action_scale, dict) else action_scale
    gains[joint] = {
      "kp": float(act["stiffness"]),
      "kd": float(act["damping"]),
      "armature": float(act["armature"]),
      "frictionloss": DEFAULT_FRICTIONLOSS if friction is None else float(friction),
      "effort_limit": float(act["effort_limit"]),
      "default_pos": float(joint_pos[joint]),
      "action_scale": float(scale),
    }
  return gains


def _write_gains_yaml(gains: dict[str, dict[str, float]], out_path: Path) -> None:
  lines = [
    "# Auto-generated by export_gains.py from config.yaml. Do not edit by hand.",
    "# Deploy contract: load by joint name (order-independent). Fields per joint:",
    f"# {', '.join(FIELDS)}",
    "",
  ]
  for joint, vals in gains.items():
    for field in FIELDS:
      lines.append(f"{joint}.{field}: {vals[field]:.6g}")
  out_path.write_text("\n".join(lines) + "\n")


def export_gains(model_dir: Path) -> Path | None:
  """Write ``<model_dir>/gains.yaml`` from ``<model_dir>/config.yaml``.

  Returns the gains.yaml path, or None if no config.yaml is present.
  """
  config_path = model_dir / "config.yaml"
  if not config_path.exists():
    logger.warning(f"skip {model_dir}: no config.yaml")
    return None
  gains = _gains_from_config(config_path)
  out_path = model_dir / "gains.yaml"
  _write_gains_yaml(gains, out_path)
  logger.info(f"wrote {out_path} ({len(gains)} joints)")
  return out_path


def main(model_dirs: tyro.conf.Positional[list[str]]) -> None:
  """Write gains.yaml into each given model directory.

  Args:
    model_dirs: one or more model directories (each containing config.yaml).
  """
  n = sum(export_gains(Path(d)) is not None for d in model_dirs)
  if n == 0:
    logger.error("no gains.yaml written")
    sys.exit(1)


if __name__ == "__main__":
  tyro.cli(main)
