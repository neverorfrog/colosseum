"""Generic policy export helpers operating on ``PolicyConfig``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, cast

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

from mjlab.tasks.velocity.rl import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)
from colosseum.deploy.config.policy import PolicyConfig
from colosseum.utils import project_root

ExporterFn = Callable[[Any, str, Any | None, str], None]
MetadataFn = Callable[[ManagerBasedRlEnv, str, str, str], None]
NormalizerFn = Callable[[Any], Any | None]


def _actor_normalizer(policy: Any) -> Any | None:
  if getattr(policy, "actor_obs_normalization", False):
    return getattr(policy, "actor_obs_normalizer", None)
  return None


@dataclass
class ExportHooks:
  """Hook functions that customize export behavior."""

  exporter_fn: ExporterFn = export_velocity_policy_as_onnx
  metadata_fn: MetadataFn | None = attach_onnx_metadata
  normalizer_fn: NormalizerFn | None = _actor_normalizer


def _resolve_path(path: Path) -> Path:
  if path.is_absolute():
    return path
  return (project_root() / path).resolve()


def build_runner(task_id: str, device: str, runner_kwargs: dict[str, Any]) -> OnPolicyRunner:
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner_cfg = asdict(rl_cfg) if is_dataclass(rl_cfg) else rl_cfg.__dict__.copy()
  runner = runner_cls(vec_env, runner_cfg, None, device, **runner_kwargs)
  return runner


def _derive_export_paths(cfg: PolicyConfig) -> tuple[Path, Path, str]:
  target_path = Path(cfg.checkpoint_path)
  if cfg.export_checkpoint_path is None:
    raise ValueError(
      "PolicyConfig.export_checkpoint_path must be provided when exporting to ONNX."
    )

  checkpoint = _resolve_path(Path(cfg.export_checkpoint_path))
  output_dir_input = cfg.export_output_dir or str(target_path.parent)
  output_dir = _resolve_path(Path(output_dir_input))
  filename = cfg.export_filename or target_path.name
  return checkpoint, output_dir, filename


def export_policy(cfg: PolicyConfig, *, hooks: ExportHooks | None = None) -> Path:
  if cfg.task_name is None:
    raise ValueError(
      "PolicyConfig.task_name must be set before exporting to ONNX."
    )

  checkpoint, output_dir, filename = _derive_export_paths(cfg)
  device = cfg.export_device or "cpu"
  runner_kwargs = (
    dict(cfg.export_runner_kwargs)
    if cfg.export_runner_kwargs is not None
    else {}
  )
  runner = build_runner(
    task_id=cfg.task_name,
    device=device,
    runner_kwargs=runner_kwargs,
  )
  env = cast(RslRlVecEnvWrapper, runner.env)
  hooks = hooks or ExportHooks()

  try:
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.alg.policy
    normalizer = hooks.normalizer_fn(policy) if hooks.normalizer_fn else None

    output_dir.mkdir(parents=True, exist_ok=True)
    hooks.exporter_fn(policy, str(output_dir), normalizer, filename)

    if hooks.metadata_fn:
      hooks.metadata_fn(env.unwrapped, cfg.export_run_path, str(output_dir), filename)

    return output_dir / filename
  finally:
    env.close()


def resolve_policy_artifact(cfg: PolicyConfig, *, hooks: ExportHooks | None = None) -> Path:
  if not cfg.use_onnx:
    return _resolve_path(Path(cfg.checkpoint_path))
  return export_policy(cfg, hooks=hooks)


__all__ = [
  "ExportHooks",
  "build_runner",
  "export_policy",
  "resolve_policy_artifact",
]
