from typing import Any, Literal
from pydantic.dataclasses import dataclass
from pydantic import Field
from colosseum.deploy.config.robot import RobotConfig

@dataclass(frozen=True)
class PolicyConfig:
    """Policy configuration.

    Specifies which policy to use and its parameters.
    """

    task_name: str = Field(
        description="Type of policy to deploy"
    )
    checkpoint_path: str = Field(
        description="Path to policy checkpoint (.pt or .onnx)"
    )
    action_scale: float = Field(
        default=0.25,
        description="Action scaling factor (must match training)"
    )
    use_onnx: bool = Field(
        default=True,
        description="Export the training checkpoint to ONNX before deployment"
    )
    export_checkpoint_path: str | None = Field(
        default=None,
        description="Source training checkpoint (.pt) to convert into ONNX"
    )
    export_output_dir: str | None = Field(
        default=None,
        description="Directory where the exported ONNX artifact will be written"
    )
    export_filename: str | None = Field(
        default=None,
        description="Filename of the exported ONNX artifact"
    )
    export_run_path: str | None = Field(
        default="local-export",
        description="Run identifier embedded in exported ONNX metadata"
    )
    export_device: str | None = Field(
        default="cpu",
        description="Device to use when instantiating the export runner"
    )
    export_runner_kwargs: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Additional keyword arguments forwarded to the export runner"
    )

    
@dataclass(frozen=True)
class VelocityCommandConfig:
    """Velocity command limits."""

    vx_max: float = Field(
        default=1.0,
        description="Maximum forward velocity (m/s)"
    )
    vy_max: float = Field(
        default=1.0,
        description="Maximum lateral velocity (m/s)"
    )
    vyaw_max: float = Field(
        default=1.0,
        description="Maximum yaw rate (rad/s)"
    )