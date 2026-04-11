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
