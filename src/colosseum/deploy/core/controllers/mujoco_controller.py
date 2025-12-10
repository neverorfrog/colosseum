from __future__ import annotations

import sys
from time import sleep
import select
import numpy as np
import torch
import mujoco
import mujoco.viewer
from booster_assets import BOOSTER_ASSETS_DIR # TODO: change to colosseum_assets

from colosseum.deploy.core.controllers.base_controller import BaseController, VelocityCommand
from colosseum.deploy.core.controllers.controller_cfg import ControllerCfg


class MujocoController(BaseController):
    def __init__(self, cfg: ControllerCfg):
        super().__init__(cfg)

        mjcf_path = self._expand_assets_placeholder(self.robot.cfg.mjcf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_model.opt.timestep = self.cfg.mujoco.physics_dt
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

    def _expand_assets_placeholder(self, path: str) -> str:
        """Replace {BOOSTER_ASSETS_DIR} placeholder in a path string.
        """
        try:
            return path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
        except Exception:
            return path

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
    def update_state(self) -> None:
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
        dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)

        base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
        base_quat = self.mj_data.sensor("orientation").data.astype(np.float32)
        base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
        base_ang_vel_b = self.mj_data.sensor("angular-velocity").data.astype(
            np.float32)

        self.robot.data.joint_pos = torch.from_numpy(dof_pos)
        self.robot.data.joint_vel = torch.from_numpy(dof_vel)
        self.robot.data.feedback_torque = torch.from_numpy(dof_torque)
        self.robot.data.root_pos_w = torch.from_numpy(base_pos_w)
        self.robot.data.root_quat_w = torch.from_numpy(base_quat)
        self.robot.data.root_lin_vel_b = torch.from_numpy(base_lin_vel_b)
        self.robot.data.root_ang_vel_b = torch.from_numpy(base_ang_vel_b)

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

        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
        kp = self.robot.joint_stiffness.numpy()
        kd = self.robot.joint_damping.numpy()
        for i in range(self.decimation):
            self.mj_data.ctrl = np.clip(
                kp * (dof_targets - dof_pos) - kd * dof_vel,
                self.mj_model.actuator_forcerange[:, 0],
                self.mj_model.actuator_forcerange[:, 1],
            )
            mujoco.mj_step(self.mj_model, self.mj_data)
            dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6:]

    def run(self):
        with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data) as viewer:

            self.viewer = viewer
            viewer.cam.elevation = -20
            if self.vel_command is not None:
                print("\nSet command (x, y, yaw): ", end="")
            self.update_state()
            self.start()
            while viewer.is_running() and self.is_running:
                sleep(self.cfg.mujoco.physics_dt * self.cfg.mujoco.decimation)
                self.update_state()
                dof_targets = self.policy_step()
                self.ctrl_step(dof_targets)
                self.viewer.cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
                self.viewer.sync()
