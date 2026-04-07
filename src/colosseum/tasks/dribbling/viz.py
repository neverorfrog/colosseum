"""Debug visualization helpers for the dribbling task."""


def draw_camera_ball_overlay(env, vis) -> None:
  """Draw head camera frame and arrow to ball in the mjlab viewer.

  No-ops silently if the d455_color camera or ball entity are absent.
  """
  try:
    cam_name = "robot/d455_color"
    cam_id = env.sim.mj_model.camera(cam_name).id
    ball = env.scene["ball"]
  except (KeyError, Exception):
    return

  env_idx = vis.env_idx
  cam_pos = env.sim.data.cam_xpos[env_idx, cam_id, :].cpu().numpy()
  # cam_xmat rows are camera local axes in world frame; transpose → columns = axes
  cam_mat = env.sim.data.cam_xmat[env_idx, cam_id, :].cpu().numpy().reshape(3, 3).T
  ball_pos = ball.data.root_link_pos_w[env_idx, :].cpu().numpy()

  vis.add_frame(position=cam_pos, rotation_matrix=cam_mat, scale=0.12, alpha=0.9)
  vis.add_arrow(start=cam_pos, end=ball_pos, color=(1.0, 0.85, 0.0, 0.8), width=0.01)
