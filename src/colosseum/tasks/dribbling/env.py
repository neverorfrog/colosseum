from dataclasses import dataclass
from typing import ClassVar

from mjlab.envs import ManagerBasedRlEnvCfg

from colosseum.envs.viewer_compatible_env import ViewerCompatibleEnv
from colosseum.tasks.dribbling.viz import draw_camera_ball_overlay


class DribblingEnv(ViewerCompatibleEnv):
  """Extends viewer compatibility with dribbling-specific camera visualization."""

  def update_visualizers(self, vis) -> None:
    super().update_visualizers(vis)
    draw_camera_ball_overlay(self, vis)


@dataclass(kw_only=True)
class DribblingEnvCfg(ManagerBasedRlEnvCfg):
  class_type: ClassVar[type] = DribblingEnv
