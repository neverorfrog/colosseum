from __future__ import annotations

import sys
from time import sleep
import select
import numpy as np
import torch
import mujoco
import mujoco.viewer

from mjlab.utils.spec import create_position_actuator

from colosseum.deploy.core.base_controller import BaseController, VelocityCommand

from colosseum.deploy.config import ControllerConfig
from colosseum.mdp.observations import compute_projected_gravity


class MujocoController(BaseController):
    def __init__(self, cfg: ControllerConfig) -> None:
        super().__init__(cfg)

        mjcf_path = self.robot.cfg.mjcf_path

        # Load spec from XML and add programmatic actuators
        # (XML actuators are commented out to avoid conflicts with training)
        spec = mujoco.MjSpec.from_file(mjcf_path)
        spec.actuators.clear()  # Ensure no XML actuators

        # Add ground plane (infinite plane at z=0)
        ground_geom = spec.worldbody.add_geom()
        ground_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground_geom.size[:] = [0, 0, 0.05]  # Infinite plane with 0.05m thickness for visuals
        ground_geom.rgba[:] = [0.5, 0.5, 0.5, 1.0]  # Gray color
        # Set ground friction (default MuJoCo friction)
        ground_geom.friction[:] = [1.0, 0.005, 0.0001]

        # Add position actuators for each joint using RobotConfig gains
        for i, joint_name in enumerate(self.robot.cfg.sim_joint_names):
            create_position_actuator(
                spec,
                joint_name,
                stiffness=self.robot.cfg.joint_stiffness[i],
                damping=self.robot.cfg.joint_damping[i],
                effort_limit=self.robot.cfg.effort_limit[i],
            )

        # Compile spec to model
        self.mj_model = spec.compile()
        self.mj_model.opt.timestep = self.cfg.physics_dt
        self.decimation = self.cfg.mujoco.decimation
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.mj_data.qpos = np.concatenate(
            [
                np.array(self.cfg.mujoco.init_pos, dtype=np.float32),
                np.array(self.cfg.mujoco.init_quat, dtype=np.float32),
                self.robot.default_joint_pos.numpy(),
            ]
        )
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_vel_command(self):
        cmd: VelocityCommand = self.vel_command
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                parts = sys.stdin.readline().strip().split()
                if len(parts) == 3:
                    (cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw) = map(float, parts)
                    print(
                        f"Updated command to: x={cmd.lin_vel_x},"
                        f"y={cmd.lin_vel_y}, yaw={cmd.ang_vel_yaw}\n"
                        "Set command (x, y, yaw): ",
                        end="",
                    )
                else:
                    raise ValueError
            except ValueError:
                print(
                    "Invalid input. Enter three numeric values. "
                    "Set command (x, y, yaw): ",
                    end="",
                )

    # TODO: make observations uniform with training pipeline
    # TODO: projected gravity. is it in mj_data?
    def update_state(self) -> None:
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
        dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)

        base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
        base_quat = self.mj_data.sensor("orientation").data.astype(np.float32)
        base_lin_vel_b = self.mj_data.sensor("imu_lin_vel").data.astype(np.float32)
        base_ang_vel_b = self.mj_data.sensor("imu_ang_vel").data.astype(np.float32)

        self.robot.data.joint_pos = torch.from_numpy(dof_pos)
        self.robot.data.joint_vel = torch.from_numpy(dof_vel)
        self.robot.data.feedback_torque = torch.from_numpy(dof_torque)
        self.robot.data.root_pos_w = torch.from_numpy(base_pos_w)
        root_quat_w = torch.from_numpy(base_quat)
        self.robot.data.root_quat_w = root_quat_w
        self.robot.data.root_lin_vel_b = torch.from_numpy(base_lin_vel_b)
        self.robot.data.root_ang_vel_b = torch.from_numpy(base_ang_vel_b)
        self.robot.data.projected_gravity_b = compute_projected_gravity(root_quat_w)

        # if self.cfg.mujoco.save_states:
        #     if not hasattr(self, 'states'):
        #         self._states = []

        #     self._states.append(np.concatenate(
        #         [base_pos_w, base_quat, dof_pos, dof_vel, dof_torque], axis=0))
        #     if len(self._states) % 100 == 0:
        #         np.savez('mujoco_states.npz', states=np.stack(self._states))
        #         print(f'saved mujoco_states.npz at {len(self._states)} steps')

    def ctrl_step(self, dof_targets: torch.Tensor):
        dof_targets = dof_targets.cpu().numpy()  # type: ignore
        if self.vel_command is not None:
            self.update_vel_command()

        # With position actuators, send position targets directly
        # MuJoCo applies internal PD control with actuator's stiffness/damping
        for _ in range(self.decimation):
            self.mj_data.ctrl = dof_targets  # Position targets, not torques!
            mujoco.mj_step(self.mj_model, self.mj_data)

    def run(self):
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:

            self.viewer: mujoco.viewer.Handle = viewer
            cam: mujoco.MjvCamera = self.viewer.cam
            cam.elevation = -20
            if self.vel_command is not None:
                print("\nSet command (x, y, yaw): ", end="")
            self.update_state()
            self.start()
            while self.viewer.is_running() and self.is_running:
                sleep(self.cfg.physics_dt * self.cfg.mujoco.decimation)
                self.update_state()
                dof_targets = self.policy_step()
                self.ctrl_step(dof_targets)
                cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
                self.viewer.sync()
