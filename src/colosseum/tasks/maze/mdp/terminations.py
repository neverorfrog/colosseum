import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactData, ContactSensor

from colosseum.tasks.dribbling.mdp.obstacle_commands import ObstacleCommand

from .observations import agent_to_goal_vector as goal_to_agent_vector
from .rewards import is_healthy


def is_not_healthy(
  env: ManagerBasedRlEnv,
  healthy_z_range: tuple[float, float] = (0.2, 1.0),
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Check if the robot is in an unhealthy state (fallen)."""
  return ~is_healthy(env, healthy_z_range, asset_cfg)


def arrived_at_goal(
  env: ManagerBasedRlEnv,
  threshold: float = 0.2,
  command_name: str = "goal",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("root_site",)),
) -> torch.Tensor:
  """Termination condition for arriving at the goal position.

  Args:
      env: The environment
      threshold: Distance threshold for arrival (meters)
      command_name: Name of the goal command term
      asset_cfg: Scene entity config for the robot

  Returns:
      Termination tensor (num_envs,) with True for terminated envs
  """
  agent_to_goal_vec = goal_to_agent_vector(env)  # [num_envs, 2]
  distance = torch.norm(agent_to_goal_vec, dim=1)  # [num_envs]
  return distance < threshold


def collided_with_wall(
  env: ManagerBasedRlEnv,
  sensor_name: str = "wall_collision",
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Termination condition for colliding with a wall with sufficient force.

  An environment is considered to have collided only if the contact force
  magnitude exceeds ``force_threshold``. The termination is additionally
  stochastic during training: each qualifying environment is independently
  terminated with probability ``env.wall_termination_prob`` (set by the
  ``wall_collision_termination_curriculum`` curriculum term). If that
  attribute is absent the probability defaults to 1.0.

  Args:
      env: The environment.
      sensor_name: Name of the contact sensor for wall collisions.
      force_threshold: Minimum contact force magnitude (N) to trigger termination.

  Returns:
      Boolean tensor (num_envs,) – True where the episode should terminate.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data: ContactData = sensor.data
  assert data.force is not None, "sensor must be configured to output force data"

  # data.force: [num_envs, N, 3] — take max force magnitude across contact slots
  force_magnitude = torch.norm(data.force, dim=-1)  # [num_envs, N]
  is_colliding = force_magnitude.amax(dim=-1) > force_threshold  # [num_envs]

  prob: float = getattr(env, "wall_termination_prob", 1.0)

  if prob <= 0.0:
    return torch.zeros_like(is_colliding)
  if prob >= 1.0:
    return is_colliding

  rand = torch.rand(env.num_envs, device=env.device)
  return is_colliding & (rand < prob)
