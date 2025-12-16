from __future__ import annotations

import sys
from time import sleep
import select
import numpy as np
import torch
import mujoco
import mujoco.viewer

from colosseum.deploy.core.base_controller import BaseController, VelocityCommand

from colosseum.deploy.config import ControllerConfig
from colosseum.mdp.observations import compute_projected_gravity


def _add_position_actuators_from_cfg(spec: mujoco.MjSpec, robot_cfg) -> None:
    """Add one position actuator per joint using RobotConfig gains/limits.

    This mirrors the mjlab helper but keeps deploy free of mjlab dependency.
    """

    if len(spec.actuators) > 0:
        return  # Actuators already present in the MJCF

    for i, joint_name in enumerate(robot_cfg.sim_joint_names):
        actuator = spec.add_actuator(name=joint_name, target=joint_name)

        # Configure as position actuator: torque = kp * (ctrl - q) - kd * qd
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

        kp = float(robot_cfg.joint_stiffness[i])
        kd = float(robot_cfg.joint_damping[i])
        actuator.gainprm[0] = kp
        actuator.biasprm[1] = -kp
        actuator.biasprm[2] = -kd

        # Allow wide ctrl range; enforce effort limits instead
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange[:] = np.array(
            [-robot_cfg.effort_limit[i], robot_cfg.effort_limit[i]],
            dtype=np.float64,
        )

        # Optional joint properties
        spec.joint(joint_name).armature = 0.3
        spec.joint(joint_name).frictionloss = 0.8


class MujocoController(BaseController):
    def __init__(self, cfg: ControllerConfig) -> None:
        super().__init__(cfg)

        mjcf_path = self.robot.cfg.mjcf_path

        # Load spec from XML and add per-joint actuators if missing
        spec = mujoco.MjSpec.from_file(mjcf_path)
        _add_position_actuators_from_cfg(spec, self.robot.cfg)

        # Add ground plane (infinite plane at z=0)
        ground_geom = spec.worldbody.add_geom()
        ground_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground_geom.size[:] = [0, 0, 0.05]  # Infinite plane with 0.05m thickness for visuals
        ground_geom.rgba[:] = [0.5, 0.5, 0.5, 1.0]  # Gray color
        # Set ground friction (default MuJoCo friction)
        ground_geom.friction[:] = [1.0, 0.005, 0.0001]

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

    # TODO: make observations uniform with training pipeline
    # TODO: projected gravity. is it in mj_data?
    def update_state(self) -> None:
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
        # Prefer actuator forces if actuators exist; otherwise use last applied torque
        if getattr(self.mj_model, "na", 0) > 0:
            dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)
        elif hasattr(self, "_last_tau"):
            dof_torque = self._last_tau.astype(np.float32)
        else:
            dof_torque = np.zeros_like(dof_pos, dtype=np.float32)

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
            self.update_command()

        if getattr(self.mj_model, "na", 0) > 0:
            # Drive actuators directly (assumes position actuators)
            for _ in range(self.decimation):
                self.mj_data.ctrl = dof_targets
                mujoco.mj_step(self.mj_model, self.mj_data)
        else:
            # Fallback: apply PD torques without actuators
            kp = np.asarray(self.robot.cfg.joint_stiffness, dtype=np.float32)
            kd = np.asarray(self.robot.cfg.joint_damping, dtype=np.float32)
            effort = np.asarray(self.robot.cfg.effort_limit, dtype=np.float32)
            q = self.mj_data.qpos.astype(np.float32)[7:]
            qd = self.mj_data.qvel.astype(np.float32)[6:]
            tau = kp * (dof_targets - q) - kd * qd
            tau = np.clip(tau, -effort, effort)
            self._last_tau = tau
            for _ in range(self.decimation):
                self.mj_data.qfrc_applied[:] = 0.0
                self.mj_data.qfrc_applied[6:] = tau
                mujoco.mj_step(self.mj_model, self.mj_data)

    def run(self):
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:

            self.viewer: mujoco.viewer.Handle = viewer
            cam: mujoco.MjvCamera = self.viewer.cam
            cam.elevation = -20
            if self.vel_command is not None and self.input_source is not None:
                print(f"\n{self.input_source.get_operation_hint()}")
            self.update_state()
            self.start()
            while self.viewer.is_running() and self.is_running:
                sleep(self.cfg.physics_dt * self.cfg.mujoco.decimation)
                self.update_state()
                dof_targets = self.policy_step()
                self.ctrl_step(dof_targets)
                cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
                self.viewer.sync()
