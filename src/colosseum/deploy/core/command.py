from abc import ABC, abstractmethod
from colosseum.deploy.config import VelocityCommandConfig

class Command(ABC):
    """Base class for command objects."""
    
    @abstractmethod
    def validate(self) -> bool:
        """Check if command is valid within limits."""
        pass

class VelocityCommand(Command):
    lin_vel_x: float
    lin_vel_y: float
    ang_vel_yaw: float

    def __init__(self, cfg: VelocityCommandConfig) -> None:
        self.vx_max = cfg.vx_max
        self.vy_max = cfg.vy_max
        self.vyaw_max = cfg.vyaw_max
        self.lin_vel_x: float = 0.0
        self.lin_vel_y: float = 0.0
        self.ang_vel_yaw: float = 0.0

    def validate(self) -> bool:
        return (abs(self.lin_vel_x) <= self.vx_max and 
                abs(self.lin_vel_y) <= self.vy_max and 
                abs(self.ang_vel_yaw) <= self.vyaw_max)