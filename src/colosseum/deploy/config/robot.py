from pydantic import Field
from pydantic.dataclasses import dataclass

@dataclass(frozen=True)
class PrepareStateConfig:
    """Safe initialization pose for real robot."""
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    joint_pos: tuple[float, ...]
    
    
@dataclass(frozen=True)
class RobotConfig:
    """Robot hardware configuration.

    Defines all robot-specific parameters including joint configuration,
    control parameters, and hardware specifications.
    """
    # Robot identity
    name: str = Field(description="Robot identifier (e.g., 'Booster_T1_23DOF')")

    # Joint configuration (immutable tuples!)
    joint_names: tuple[str, ...] = Field(
        description="Joint names in real robot hardware order"
    )
    sim_joint_names: tuple[str, ...] = Field(
        description="Joint names in simulation order (alphabetical)"
    )

    # Body configuration
    body_names: tuple[str, ...] = Field(
        description="Body names in real robot order"
    )
    sim_body_names: tuple[str, ...] = Field(
        description="Body names in simulation order"
    )

    # Control parameters (MUST match training!)
    joint_stiffness: tuple[float, ...] = Field(
        description="PD gains (Kp) - must match training values exactly"
    )
    joint_damping: tuple[float, ...] = Field(
        description="PD gains (Kd) - must match training values exactly"
    )
    joint_armature: tuple[float, ...] = Field(
        description="Joint armature (reflected inertia) - must match training XML"
    )
    default_joint_pos: tuple[float, ...] = Field(
        description="Default standing pose (joint positions in radians)"
    )
    effort_limit: tuple[float, ...] = Field(
        description="Maximum torque limits per joint (Nm)"
    )

    # Required hardware-specific
    mjcf_path: str = Field(
        description="Path to MuJoCo MJCF model file"
    )
    prepare_state: PrepareStateConfig = Field(
        description="Safe initialization pose configuration"
    )

    # Optional hardware-specific (must come after required fields!)
    parallel_joint_indices: tuple[int, ...] = Field(
        default=(),
        description="Indices of mechanically coupled joints"
    )

    @property
    def num_joints(self) -> int:
        """Number of controlled joints."""
        return len(self.joint_names)

    @property
    def num_bodies(self) -> int:
        """Number of bodies."""
        return len(self.body_names)
    
    def __post_init__(self):
        assert (
            len(self.joint_names)
            == len(self.joint_stiffness)
            == len(self.joint_damping)
            == len(self.joint_armature)
            == len(self.default_joint_pos)
            == len(self.effort_limit)
        )