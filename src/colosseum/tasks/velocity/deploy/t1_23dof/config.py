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

_DEFAULT_POLICY_PATH = (
    project_root()
    / "src"
    / "colosseum"
    / "tasks"
    / "velocity"
    / "deploy"
    / "t1_23dof"
    / "models"
    / "policy.onnx"
)

_TRAINING_CHECKPOINT_PATH = (
    project_root() / "wandb" / "run-20251209_112107-dtxd5qxc" / "files" / "model_28500.pt"
)

# Default T1 velocity deployment configuration
T1_23DOF_VELOCITY = ControllerConfig(
    # Policy execution frequency (50Hz)
    policy_dt=0.02,
    # Robot configuration (centralized T1 specs)
    robot=T1_23DOF_ROBOT_CFG,
    # Policy configuration
    policy=PolicyConfig(
        task_name="Velocity-Rough-Booster-T1",
        checkpoint_path=str(_DEFAULT_POLICY_PATH),
        action_scale=0.25,  # Must match training config
        export_checkpoint_path=str(_TRAINING_CHECKPOINT_PATH),
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
        init_pos=(0.0, 0.0, 0.6),  # Start at 60cm height
        init_quat=(1.0, 0.0, 0.0, 0.0),  # Upright orientation (w, x, y, z)
        decimation=10,  # 500Hz physics / 50Hz policy
        save_states=False,  # Disable state logging by default
    ),
    # Booster robot parameters
    booster=BoosterConfig(
        low_state_dt=0.002,  # 500Hz sensor updates
        metrics_max_events=2000,  # Metrics buffer size
    ),
)
