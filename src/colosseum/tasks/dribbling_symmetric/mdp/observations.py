import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

from colosseum.mdp.ball_rewards import camera_fov_mask as _camera_fov_mask
from colosseum.tasks.dribbling.mdp.obstacle_commands import ObstacleCommand


_PARK_FAR: float = 1000.0


def ball_position(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position relative to robot base in robot body frame. Shape (N, 2)."""
  robot = env.scene["robot"]
  ball = env.scene["ball"]

  relative_w = ball.data.root_link_pos_w[:, :3] - robot.data.root_link_pos_w[:, :3]
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  return quat_apply(quat_conj, relative_w)[:, :2]


def ball_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball linear velocity in world frame. Shape (N, 3)."""
  return env.scene["ball"].data.root_link_lin_vel_w[:, :3]


def ball_vel_command_body(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Desired ball velocity command rotated into robot body frame. Shape (N, 3).

  The command manager stores the target in world frame. This observation
  rotates it into the robot's body frame so the actor sees a frame-consistent
  view: body-frame command + body-frame ball state from the encoder.
  Rewards remain world-frame internally and are unaffected.
  """
  robot = env.scene["robot"]
  cmd_w = env.command_manager.get_command(command_name)  # (N, 3) world
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  return quat_apply(quat_conj, cmd_w)


def ball_velocity_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball linear velocity XY in robot body frame. Shape (N, 2)."""
  robot = env.scene["robot"]
  ball = env.scene["ball"]
  vel_w = ball.data.root_link_lin_vel_w[:, :3]
  quat_w = robot.data.root_link_quat_w
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  return quat_apply(quat_conj, vel_w)[:, :2]


def base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Robot base height above ground. Shape (N, 1)."""
  return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def ball_mass(env: ManagerBasedRlEnv, ball_mass: float) -> torch.Tensor:
  """Ball mass as privileged scalar. Shape (N, 1)."""
  N = env.num_envs
  return torch.full((N, 1), ball_mass, device=env.device, dtype=torch.float32)


def ball_friction(env: ManagerBasedRlEnv, ball_friction: float) -> torch.Tensor:
  """Ball sliding friction as privileged scalar. Shape (N, 1)."""
  N = env.num_envs
  return torch.full((N, 1), ball_friction, device=env.device, dtype=torch.float32)


def foot_ball_contact_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Net contact force magnitude between feet and ball. Shape (N, 1).

  Returns the norm of the XYZ net force so the policy knows how hard
  it is currently hitting the ball.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  force = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return force.norm(dim=-1, keepdim=True)


def obstacle_position_b(
  env: ManagerBasedRlEnv,
  camera_name: str = "robot/d455_color",
  camera_fovy: float = 60.0,
  camera_aspect_ratio: float = 4.0 / 3.0,
  depth_clip: float = 6.0,
) -> torch.Tensor:
  """Closest in-FOV obstacle XY position in robot body frame.

  Returns (N, 2). Obstacles outside the camera frustum are treated as
  inactive and replaced with the far-away sentinel so the policy only
  reacts to what the camera can see.
  """
  robot = env.scene["robot"]
  N = env.num_envs

  quat_w = robot.data.root_link_quat_w  # (N, 4)
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  robot_pos_w = robot.data.root_link_pos_w[:, :2]  # (N, 2)
  zeros = torch.zeros(N, 1, device=quat_w.device)
  term: ObstacleCommand = env.command_manager.get_term("adversary")

  if term.cfg.num_active == 0:
    nearest_xy = torch.full((N, 2), _PARK_FAR, device=quat_w.device)
  else:
    active_xy = term.obstacle_positions_w[:, : term.cfg.num_active]  # (N, Ka, 2)
    dist = (active_xy - robot_pos_w.unsqueeze(1)).norm(dim=-1)  # (N, Ka)
    idx = dist.argmin(dim=-1)
    nearest_xy = active_xy[torch.arange(N, device=env.device), idx]  # (N, 2)
    in_fov = _camera_fov_mask(env, nearest_xy, camera_name, camera_fovy, camera_aspect_ratio, depth_clip)
    sentinel = torch.full((N, 2), _PARK_FAR, device=quat_w.device)
    nearest_xy = torch.where(in_fov.unsqueeze(-1), nearest_xy, sentinel)

  rel_xy = nearest_xy - robot_pos_w
  rel_3d = torch.cat([rel_xy, zeros], dim=-1)
  return quat_apply(quat_conj, rel_3d)[:, :2]  # (N, 2)


def obstacle_velocity_b(
  env: ManagerBasedRlEnv,
  camera_name: str = "robot/d455_color",
  camera_fovy: float = 60.0,
  camera_aspect_ratio: float = 4.0 / 3.0,
  depth_clip: float = 6.0,
) -> torch.Tensor:
  """Closest in-FOV obstacle XY velocity in robot body frame.

  Returns (N, 2). Obstacles outside the camera frustum are masked to zero
  velocity, consistent with the position sentinel treatment.
  """
  robot = env.scene["robot"]
  N = env.num_envs

  quat_w = robot.data.root_link_quat_w  # (N, 4)
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  zeros = torch.zeros(N, 1, device=quat_w.device)
  term: ObstacleCommand = env.command_manager.get_term("adversary")

  if term.cfg.num_active == 0:
    nearest_vel_w = torch.zeros((N, 2), device=quat_w.device)
  else:
    robot_pos_w = robot.data.root_link_pos_w[:, :2]  # (N, 2)
    active_xy = term.obstacle_positions_w[:, : term.cfg.num_active]  # (N, Ka, 2)
    dist = (active_xy - robot_pos_w.unsqueeze(1)).norm(dim=-1)  # (N, Ka)
    idx = dist.argmin(dim=-1)
    nearest_xy = active_xy[torch.arange(N, device=env.device), idx]  # (N, 2)
    nearest_vel_w = term.obstacle_velocities_w[torch.arange(N, device=env.device), idx]
    in_fov = _camera_fov_mask(env, nearest_xy, camera_name, camera_fovy, camera_aspect_ratio, depth_clip)
    nearest_vel_w = torch.where(in_fov.unsqueeze(-1), nearest_vel_w, torch.zeros_like(nearest_vel_w))

  vel_3d = torch.cat([nearest_vel_w, zeros], dim=-1)
  return quat_apply(quat_conj, vel_3d)[:, :2]  # (N, 2)
