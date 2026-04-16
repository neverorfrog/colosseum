import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply


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


def obstacle_positions_b(
  env: ManagerBasedRlEnv,
  num_obstacles: int = 1,
) -> torch.Tensor:
  """Obstacle XY positions in robot body frame, read directly from MuJoCo.

  Returns a flat (N, num_obstacles * 2) tensor ordered as [x0, y0, x1, y1, ...].
  Inactive obstacles are parked far away so they appear hundreds of metres away —
  naturally near-zero danger in the encoder gate.
  """
  robot = env.scene["robot"]
  N = env.num_envs

  quat_w = robot.data.root_link_quat_w  # (N, 4)
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  robot_pos_w = robot.data.root_link_pos_w[:, :2]  # (N, 2)
  zeros = torch.zeros(N, 1, device=quat_w.device)

  parts: list[torch.Tensor] = []
  for k in range(num_obstacles):
    obs_pos_w = env.scene[f"obstacle_{k}"].data.root_link_pos_w[:, :2]  # (N, 2)
    rel_xy = obs_pos_w - robot_pos_w
    rel_3d = torch.cat([rel_xy, zeros], dim=-1)
    parts.append(quat_apply(quat_conj, rel_3d)[:, :2])  # (N, 2)

  return torch.cat(parts, dim=-1)  # (N, num_obstacles*2)


def obstacle_velocities_b(
  env: ManagerBasedRlEnv,
  num_obstacles: int = 1,
) -> torch.Tensor:
  """Obstacle XY velocities in robot body frame, read directly from MuJoCo.

  Returns a flat (N, num_obstacles * 2) tensor ordered as [vx0, vy0, vx1, vy1, ...].
  Velocity is a pure rotation from world into body frame (no translation).
  """
  robot = env.scene["robot"]
  N = env.num_envs

  quat_w = robot.data.root_link_quat_w  # (N, 4)
  quat_conj = torch.cat([quat_w[:, :1], -quat_w[:, 1:]], dim=-1)
  zeros = torch.zeros(N, 1, device=quat_w.device)

  parts: list[torch.Tensor] = []
  for k in range(num_obstacles):
    vel_w = env.scene[f"obstacle_{k}"].data.root_link_lin_vel_w[:, :2]  # (N, 2)
    vel_3d = torch.cat([vel_w, zeros], dim=-1)
    parts.append(quat_apply(quat_conj, vel_3d)[:, :2])  # (N, 2)

  return torch.cat(parts, dim=-1)  # (N, num_obstacles*2)
