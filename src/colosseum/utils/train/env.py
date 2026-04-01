"""Environment utilities for training and evaluation."""

import torch
from mjlab.envs import ManagerBasedRlEnv


class ViewerCompatibleEnv(ManagerBasedRlEnv):
  """Adds get_observations() for mjlab viewer compatibility.

  The mjlab viewer expects environments to have a get_observations() method,
  but ManagerBasedRlEnv doesn't provide it.
  """

  def get_observations(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Get current observations without stepping the environment."""
    return self.observation_manager.compute()

  def update_visualizers(self, vis) -> None:
    """Extend base debug viz with camera-specific overlays for dribbling.

    Calls super() first so manager_visualizers (e.g. BallVelocityCommand arrow)
    and sensor debug vis are preserved. Then adds, if available:
      - Camera coordinate frame (RGB axes) at the head camera position
      - Yellow arrow from camera to ball
    Toggle with 'R' in the viewer.
    TODO: move this elsewhere
    """
    super().update_visualizers(vis)

    try:
      cam_name = "robot/d455_color"
      cam_id = self.sim.mj_model.camera(cam_name).id
      ball = self.scene["ball"]
    except (KeyError, Exception):
      return

    env_idx = vis.env_idx
    cam_pos = self.sim.data.cam_xpos[env_idx, cam_id, :].cpu().numpy()
    # cam_xmat rows are camera local axes in world frame; transpose → columns = axes
    cam_mat = self.sim.data.cam_xmat[env_idx, cam_id, :].cpu().numpy().reshape(3, 3).T
    ball_pos = ball.data.root_link_pos_w[env_idx, :].cpu().numpy()

    vis.add_frame(position=cam_pos, rotation_matrix=cam_mat, scale=0.12, alpha=0.9)
    vis.add_arrow(start=cam_pos, end=ball_pos, color=(1.0, 0.85, 0.0, 0.8), width=0.01)
