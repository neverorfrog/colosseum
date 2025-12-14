from pydantic.dataclasses import dataclass
from pydantic import Field

@dataclass(frozen=True)
class MujocoConfig:
    """MuJoCo simulation parameters."""

    init_pos: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.6),
        description="Initial base position (x, y, z)"
    )
    init_quat: tuple[float, float, float, float] = Field(
        default=(1.0, 0.0, 0.0, 0.0),
        description="Initial orientation quaternion (w, x, y, z)"
    )
    decimation: int = Field(
        default=10,
        description="Physics steps per policy step"
    )
    save_states: bool = Field(
        default=False,
        description="Enable state logging for debugging"
    )
    
@dataclass(frozen=True)
class BoosterConfig:
    """Booster robot-specific parameters."""

    low_state_dt: float = Field(
        default=0.002,
        description="ROS2 low_state update period (500Hz = 0.002s)"
    )
    metrics_max_events: int = Field(
        default=2000,
        description="Maximum events in metrics buffer"
    )