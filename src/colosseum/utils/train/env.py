"""Environment factory for training and evaluation."""

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

from colosseum.envs.viewer_compatible_env import ViewerCompatibleEnv


def make_env(env_cfg: ManagerBasedRlEnvCfg, device: str, render_mode: str | None = None) -> ManagerBasedRlEnv:
  """Create an environment from cfg, using cfg.class_type if set, else ViewerCompatibleEnv."""
  env_class = getattr(env_cfg, "class_type", None) or ViewerCompatibleEnv
  return env_class(cfg=env_cfg, device=device, render_mode=render_mode)
