"""T1 velocity tracking deployment preset (flat terrain).

The checkpoint path points to the stable symlink created by export-onnx:
    pixi run -e train export-onnx task:t1-velocity-flat
    → models/t1-velocity-flat_ppo_latest.onnx
"""

from colosseum.deploy.config import (
  BoosterConfig,
  ControllerConfig,
  MujocoConfig,
  PolicyConfig,
  VelocityCommandConfig,
)
from colosseum.deploy.input import InputConfig
from colosseum.robots.t1_23dof.deploy import T1_23DOF_ROBOT_CFG
from colosseum.utils import project_root

_ONNX_DIR = project_root() / "models"

T1_23DOF_VELOCITY_FLAT = ControllerConfig(
  policy_dt=0.02,
  robot=T1_23DOF_ROBOT_CFG,
  policy=PolicyConfig(
    task_name="t1-velocity-flat",
    checkpoint_path=str(_ONNX_DIR / "t1-velocity-flat_ppo_latest.onnx"),
    action_scale=0.25,
    use_onnx=False,
  ),
  vel_command=VelocityCommandConfig(
    vx_max=1.0,
    vy_max=1.0,
    vyaw_max=1.0,
  ),
  input=InputConfig("joystick"),
  mujoco=MujocoConfig(
    init_pos=(0.0, 0.0, 0.665),
    init_quat=(1.0, 0.0, 0.0, 0.0),
    decimation=4,
    save_states=False,
  ),
  booster=BoosterConfig(
    low_state_dt=0.002,
    metrics_max_events=2000,
  ),
)
