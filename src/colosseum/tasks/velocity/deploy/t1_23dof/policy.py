"""T1 23-DOF velocity tracking policy for deployment.

This policy computes observations from sensor data following the
VelocityObservationSpec contract defined in tasks/velocity/mdp/observation_spec.py
"""

from pathlib import Path
import torch

from colosseum.deploy.core.controllers import BaseController, Policy
from colosseum.tasks.velocity.mdp.observation_spec import VELOCITY_OBS_SPEC
from colosseum.deploy.core.controllers import PolicyCfg


class T1VelocityPolicy(Policy):
    """Velocity tracking policy for T1 23-DOF deployment.

    Computes observations from sensor data to match training specification.
    """

    def __init__(self, checkpoint_path: str, controller: BaseController):
        super().__init__(PolicyCfg(), controller)  # TODO: policy cfg needed
        self.robot = controller.robot
        self.vel_command = controller.vel_command

        # Load TorchScript model
        model_path = Path(checkpoint_path)
        if not model_path.is_absolute():
            # Relative to this file
            model_path = Path(__file__).parent / model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[T1VelocityPolicy] Loading model from: {model_path}")
        self._model: torch.jit.ScriptModule = torch.jit.load(str(model_path))
        self._model.eval()

        # Action scaling (must match training!)
        # scale = action_scale_factor * effort_limit / stiffness
        self.action_scale_factor = 0.25  # From training config
        self.action_scale = (
            self.action_scale_factor
            * self.robot.effort_limit
            / self.robot.joint_stiffness
        )

        # Verify observation size
        num_joints = self.robot.num_joints
        expected_obs_size = VELOCITY_OBS_SPEC.compute_size(num_joints)
        print(f"[T1VelocityPolicy] Robot: {self.robot.cfg.name}")
        print(f"[T1VelocityPolicy] Joints: {num_joints}")
        print(f"[T1VelocityPolicy] Expected observation size: {expected_obs_size}")
        print(VELOCITY_OBS_SPEC.describe(num_joints))

    def reset(self) -> None:
        """Reset policy state at start of episode."""
        self.last_action = torch.zeros(self.robot.num_joints, dtype=torch.float32)
        print("[T1VelocityPolicy] Reset complete")

    def compute_observation(self) -> torch.Tensor:
        """Compute observations from sensor data.

        Returns:
            Observation tensor of shape (1, obs_size) matching VelocityObservationSpec.
        """
        # Get joint mapping (real robot order → simulation order)
        real2sim_map = self.robot.data.real2sim_joint_indexes

        # Velocity commands (from command interface or VelocityCommand object)
        vel_cmd = torch.tensor(
            [
                self.vel_command.lin_vel_x,
                self.vel_command.lin_vel_y,
                self.vel_command.ang_vel_yaw,
            ],
            dtype=torch.float32,
        )

        # Base angular velocity (from IMU)
        base_ang_vel = self.robot.data.root_ang_vel_b

        # Projected gravity (computed by controller from IMU)
        projected_gravity = self.robot.data.projected_gravity_b

        # Joint positions relative to default (in simulation order!)
        joint_pos = self.robot.data.joint_pos[real2sim_map]
        default_pos = self.robot.default_joint_pos[real2sim_map]
        joint_pos_rel = joint_pos - default_pos

        # Joint velocities (in simulation order!)
        joint_vel = self.robot.data.joint_vel[real2sim_map]

        # Last action (already in simulation order from previous step)
        last_action = self.last_action

        # Concatenate following VelocityObservationSpec.ORDER
        obs = torch.cat(
            [
                vel_cmd,           # (3,)
                base_ang_vel,      # (3,)
                projected_gravity, # (3,)
                joint_pos_rel,     # (23,)
                joint_vel,         # (23,)
                last_action,       # (23,)
            ],
            dim=-1,
        )

        # Validate
        VELOCITY_OBS_SPEC.validate_observation(obs, self.robot.num_joints)

        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        """Run policy inference and return joint targets.

        Returns:
            Joint position targets in real robot order (23,).
        """
        with torch.no_grad():
            # Compute observations
            obs = self.compute_observation()

            # Run model (returns normalized actions in simulation order)
            action = self._model(obs).flatten()

            # Store for next step (keep in simulation order)
            self.last_action = action

            # Map from simulation order to real robot order
            sim2real_map = self.robot.data.sim2real_joint_indexes

            # Scale actions and add default positions
            joint_targets = (
                action[sim2real_map] * self.action_scale + self.robot.default_joint_pos
            )

            return joint_targets
