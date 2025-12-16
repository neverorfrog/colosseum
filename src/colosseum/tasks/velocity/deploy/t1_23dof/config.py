"""T1 23-DOF velocity tracking deployment preset.

Complete deployment configuration for running velocity tracking on the T1 robot.
"""

from colosseum.deploy.config import (
    BoosterConfig,
    ControllerConfig,
    MujocoConfig,
    PolicyConfig,
    VelocityCommandConfig,
)
from colosseum.deploy.input import InputConfig
from colosseum.robots.t1_23dof.deploy_config import T1_23DOF_ROBOT_CFG
from colosseum.utils import project_root

_DEFAULT_POLICY_PATH_ROUGH = (
    project_root()
    / "src"
    / "colosseum"
    / "tasks"
    / "velocity"
    / "deploy"
    / "t1_23dof"
    / "models"
    / "policy_rough.onnx"
)

_TRAINING_CHECKPOINT_PATH_ROUGH = (
    project_root() / "wandb" / "run-20251209_112107-dtxd5qxc" / "files" / "model_28500.pt"
)

# Default T1 velocity deployment configuration
T1_23DOF_VELOCITY_ROUGH = ControllerConfig(
    # Policy execution frequency (50Hz)
    policy_dt=0.02,
    # Robot configuration (centralized T1 specs)
    robot=T1_23DOF_ROBOT_CFG,
    # Policy configuration
    policy=PolicyConfig(
        task_name="Velocity-Rough-Booster-T1",
        checkpoint_path=str(_DEFAULT_POLICY_PATH_ROUGH),
        action_scale=0.25,  # Must match training config
        use_onnx=False,  # Pre-exported ONNX provided; avoid mjlab dependency during deploy
    ),
    # Velocity command limits
    vel_command=VelocityCommandConfig(
        vx_max=1.0,  # Max forward velocity (m/s)
        vy_max=0.5,  # Max lateral velocity (m/s)
        vyaw_max=1.0,  # Max yaw rate (rad/s)
    ),
    input=InputConfig('joystick'),
    # MuJoCo simulation parameters
    mujoco=MujocoConfig(
        init_pos=(0.0, 0.0, 0.665),  # Using training default height
        init_quat=(1.0, 0.0, 0.0, 0.0),  # Upright orientation (w, x, y, z)
        decimation=4,  # FIXED: Match training (200Hz physics @ 50Hz policy, dt=0.005s)
        save_states=False,  # Disable state logging by default
    ),
    # Booster robot parameters
    booster=BoosterConfig(
        low_state_dt=0.002,  # 500Hz sensor updates
        metrics_max_events=2000,  # Metrics buffer size
    ),
)


_DEFAULT_POLICY_PATH_FLAT = (
    project_root()
    / "src"
    / "colosseum"
    / "tasks"
    / "velocity"
    / "deploy"
    / "t1_23dof"
    / "models"
    / "policy_flat.onnx"
)

_TRAINING_CHECKPOINT_PATH_FLAT = (
    project_root() / "wandb" / "run-20251214_194727-7w8taflc" / "files" / "model_2100.pt"
)

# Default T1 velocity deployment configuration
T1_23DOF_VELOCITY_FLAT = ControllerConfig(
    # Policy execution frequency (50Hz)
    policy_dt=0.02,
    # Robot configuration (centralized T1 specs)
    robot=T1_23DOF_ROBOT_CFG,
    # Policy configuration
    policy=PolicyConfig(
        task_name="Velocity-Flat-Booster-T1",
        checkpoint_path=str(_DEFAULT_POLICY_PATH_FLAT),
        action_scale=0.25,  # Must match training config
        use_onnx=False,  # Pre-exported ONNX provided; avoid mjlab dependency during deploy
    ),
    # Velocity command limits
    vel_command=VelocityCommandConfig(
        vx_max=1.0,  # Max forward velocity (m/s)
        vy_max=0.5,  # Max lateral velocity (m/s)
        vyaw_max=1.0,  # Max yaw rate (rad/s)
    ),
    input=InputConfig('joystick'),
    # MuJoCo simulation parameters
    mujoco=MujocoConfig(
        init_pos=(0.0, 0.0, 0.665),  # Using training default height
        init_quat=(1.0, 0.0, 0.0, 0.0),  # Upright orientation (w, x, y, z)
        decimation=4,  # FIXED: Match training (200Hz physics @ 50Hz policy, dt=0.005s)
        save_states=False,  # Disable state logging by default
    ),
    # Booster robot parameters
    booster=BoosterConfig(
        low_state_dt=0.002,  # 500Hz sensor updates
        metrics_max_events=2000,  # Metrics buffer size
    ),
)

