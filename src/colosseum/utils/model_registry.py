"""Model registry for exported ONNX policies.

Managed file: models/registry.yaml

Every export is registered. The per-task "default" determines which model
arena loads via models/<task>/default.onnx symlink.

Directory layout:
  models/<task>/
    <name>/                    # per-model subfolder
      <task>_<algo>_<name>.onnx
      <task>_<algo>_<name>.pt
    default.onnx  -> <name>/<task>_<algo>_<name>.onnx
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

REGISTRY_PATH = Path("models") / "registry.yaml"


def _load() -> dict:
  if REGISTRY_PATH.exists():
    return yaml.safe_load(REGISTRY_PATH.read_text()) or {}
  return {}


def _save(data: dict) -> None:
  REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
  REGISTRY_PATH.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


class ModelRegistry:
  """Read/write the models/registry.yaml file.

  On-disk format:
      <task_name>:
        default: <name>
        models:
          <name>:
            file: <subdir/filename>   # relative to models/<task_name>/
            run: <run_directory>
            step: <global_step>
            created: <iso_timestamp>
  """

  @staticmethod
  def register(
    task: str,
    name: str,
    file: str,
    run: str,
    step: int,
  ) -> None:
    """Register a model. Sets as default if none exists yet."""
    data = _load()
    task_data = data.setdefault(task, {})
    models = task_data.setdefault("models", {})

    models[name] = {
      "file": file,
      "run": run,
      "step": step,
      "created": datetime.now(timezone.utc).isoformat(),
    }

    if "default" not in task_data:
      task_data["default"] = name
      logger.info(f"Set default for '{task}' -> '{name}' (first registered)")

    _save(data)
    _sync_default_symlink(task, name, models[name])
    logger.info(f"Registered '{name}' for task '{task}'")

  @staticmethod
  def get(task: str, name: str) -> dict | None:
    data = _load()
    return data.get(task, {}).get("models", {}).get(name)

  @staticmethod
  def get_default(task: str) -> dict | None:
    data = _load()
    task_data = data.get(task, {})
    default_name = task_data.get("default")
    if default_name:
      return task_data.get("models", {}).get(default_name)
    return None

  @staticmethod
  def set_default(task: str, name: str) -> None:
    data = _load()
    if task not in data:
      raise KeyError(f"Task '{task}' not in registry")
    if name not in data[task].get("models", {}):
      raise KeyError(f"Model '{name}' not found for task '{task}'")
    data[task]["default"] = name
    _save(data)
    _sync_default_symlink(task, name, data[task]["models"][name])

  @staticmethod
  def remove(task: str, name: str, delete_files: bool = False) -> None:
    data = _load()
    task_data = data.get(task)
    if not task_data:
      raise KeyError(f"Task '{task}' not in registry")
    if name not in task_data.get("models", {}):
      raise KeyError(f"Model '{name}' not found for task '{task}'")

    entry = task_data["models"].pop(name)

    if delete_files:
      _delete_model_subdir(task, entry.get("file", ""))

    if task_data.get("default") == name:
      remaining = list(task_data["models"])
      if remaining:
        task_data["default"] = remaining[0]
        logger.info(f"Default for '{task}' moved to '{remaining[0]}'")
        _sync_default_symlink(task, remaining[0], task_data["models"][remaining[0]])
      else:
        task_data.pop("default", None)
        _clear_default_symlink(task)
        if not task_data["models"]:
          del data[task]

    _save(data)
    logger.info(f"Removed '{name}' from task '{task}'")

  @staticmethod
  def list_models(task: str) -> list[dict]:
    data = _load()
    task_data = data.get(task, {})
    default_name = task_data.get("default")
    models = task_data.get("models", {})
    return [
      {"name": name, "default": (name == default_name), **info}
      for name, info in models.items()
    ]


# ------------------------------------------------------------------
# Symlink & file helpers
# ------------------------------------------------------------------

def _sync_default_symlink(task: str, name: str, entry: dict) -> None:
  task_dir = Path("models") / task
  task_dir.mkdir(parents=True, exist_ok=True)
  link = task_dir / "default.onnx"
  _link_to(link, entry["file"])


def _clear_default_symlink(task: str) -> None:
  link = Path("models") / task / "default.onnx"
  if link.is_symlink() or link.exists():
    link.unlink()


def _link_to(link: Path, target: str) -> None:
  if link.is_symlink() or link.exists():
    link.unlink()
  link.symlink_to(target)


def _delete_model_subdir(task: str, file_path: str) -> None:
  if not file_path:
    return
  subdir = Path("models") / task / Path(file_path).parent
  if subdir.exists() and subdir.is_dir():
    shutil.rmtree(subdir)
    logger.info(f"Deleted: {subdir}")
