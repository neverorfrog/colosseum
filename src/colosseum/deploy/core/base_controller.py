from __future__ import annotations
from abc import abstractmethod
import torch

from colosseum.deploy.config import ControllerConfig
from colosseum.deploy.core.registry import TASK_REGISTRY
from colosseum.deploy.core.policy import Policy
from colosseum.deploy.core.robot import BoosterRobot
from colosseum.deploy.core.command import VelocityCommand
from colosseum.deploy.input import BaseInputSource, create_input_source

class BaseController:
    """Simple deployment environment skeleton and execution overview.

    This class provides a minimal, dependency-light interface suitable for
    deployment scripts and controllers. It defines the method contract used by
    concrete controller implementations and documents the typical runtime
    execution order.

    Public method contract
    - `start(initial_state=None) -> obs`: prepare controller and policy for
        execution and return initial observation.
    - `policy_step() -> torch.Tensor`: invoke policy inference for one step
        and return the action tensor.
    - `ctrl_step(dof_targets: torch.Tensor) -> None`: apply action to the
        environment (send to actuators / shared buffer / simulator).
    - `update_state() -> None`: refresh internal robot state from sensors or
        shared buffers (called each control loop iteration before inference).
    - `stop() -> None`: stop the running session; should be idempotent.
    - `run() -> None`: high-level entry point for a controller process or
        thread (optional to implement for each concrete controller).

    Concrete controllers may implement `run()` to orchestrate the typical
    execution flow below:

        start()

            |
            v
    +----------------- main loop -----------------+
    |  update_state()                                |
    |      |                                         |
    |      v                                         |
    |  policy_step()  -> (action tensor)            |
    |      |                                         |
    |      v                                         |
    |  ctrl_step(action)                             |
    |      |                                         |
    +-----------------------------------------------+
            |
            v
            stop() -> cleanup()/finalize()

    Notes and recommendations
    - `update_state()` should read the latest sensor/shared-buffer data and
        populate `self.robot.data` before `policy_step()` is called.
    - `policy_step()` is responsible only for producing actions and should
        not have side-effects that interfere with `update_state()`.
    - `ctrl_step()` applies the action produced by the policy to actuators or
        publish it.
    """

    cfg: ControllerConfig
    robot: BoosterRobot
    vel_command: VelocityCommand
    policy: Policy

    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg
        self._step_count: int = 0
        self._elapsed_s: float = 0.0
        self.is_running: bool = False
        self.robot = BoosterRobot(cfg.robot)
        self.vel_command = None  # type: ignore
        self.input_source: BaseInputSource | None = None
        if self.cfg.vel_command is not None and self.cfg.input is not None:
            self.vel_command = VelocityCommand(self.cfg.vel_command)
            self.input_source = create_input_source(self.cfg.input)

        # Get policy class from registry and instantiate
        policy_class = TASK_REGISTRY.get_policy(cfg.policy.task_name)
        self.policy = policy_class(self)  # Direct instantiation

    def start(self):
        """Begin a deployment session.
        """
        self._step_count = 0
        self._elapsed_s = 0.0
        self.is_running = True
        self.policy.reset()

    def policy_step(self) -> torch.Tensor:
        """Execute one inference step and return the action.

        Returns:
            action tensor
        """
        if not self.is_running:
            raise RuntimeError("Environment.step() called before start().")

        self._step_count += 1
        self._elapsed_s = self._step_count * self.cfg.policy_dt

        return self.policy.inference()

    def stop(self) -> None:
        """Stop and clean up the deployment session."""
        self.is_running = False
        if self.input_source is not None:
            self.input_source.close()
            
    def update_command(self) -> None:
        """Default implementation using input_source."""
        if self.input_source is None or self.vel_command is None:
            return

        vx_norm = self.input_source.get_vx_cmd()
        vy_norm = self.input_source.get_vy_cmd()
        vyaw_norm = self.input_source.get_vyaw_cmd()

        self.vel_command.lin_vel_x = vx_norm * self.vel_command.vx_max
        self.vel_command.lin_vel_y = vy_norm * self.vel_command.vy_max
        self.vel_command.ang_vel_yaw = vyaw_norm * self.vel_command.vyaw_max

    @abstractmethod
    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        """Advance the environment by one control step.

        Args:
            dof_targets: Action tensor for this step (dof targets).
        """

    @abstractmethod
    def update_state(self) -> None:
        """Update robot data from sensors or shared buffers."""

    @abstractmethod
    def run(self) -> None:
        """Main loop entry point."""
