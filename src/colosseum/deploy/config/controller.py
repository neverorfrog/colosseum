from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import Field, computed_field

from .robot import RobotConfig
from .policy import PolicyConfig, VelocityCommandConfig
from .backend import MujocoConfig, BoosterConfig
from colosseum.deploy.input import InputConfig

@dataclass(frozen=True)
class ControllerConfig:
    """Top-level deployment configuration.

    Combines all configuration components for a complete deployment.
    """

    # Core parameters
    policy_dt: float = Field(
        default=0.02,
        description="Policy execution frequency (s). Default 0.02s = 50Hz"
    )

    # Required sub-configs
    robot: RobotConfig = Field(
        description="Robot hardware configuration"
    )
    policy: PolicyConfig = Field(
        description="Policy configuration"
    )

    # Optional sub-configs
    vel_command: Optional[VelocityCommandConfig] = Field(
        default=None,
        description="Velocity command configuration (for velocity tasks)"
    )
    input: Optional[InputConfig] = Field(
        default=None,
        description="Input source configuration"
    )

    # Backend-specific configs
    mujoco: MujocoConfig = Field(
        default_factory=MujocoConfig,
        description="MuJoCo simulation parameters"
    )
    booster: BoosterConfig = Field(
        default_factory=BoosterConfig,
        description="Booster robot parameters"
    )

    @computed_field
    @property
    def physics_dt(self) -> float:
        """Computed physics timestep."""
        return self.policy_dt / self.mujoco.decimation