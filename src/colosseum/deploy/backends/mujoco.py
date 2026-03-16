from __future__ import annotations

from time import sleep
import numpy as np
import torch
import mujoco
import mujoco.viewer

from colosseum.deploy.core.base_controller import BaseController

from colosseum.deploy.config import ControllerConfig
from colosseum.mdp.observations import compute_projected_gravity
from colosseum.deploy.config import RobotConfig

class MujocoController(BaseController):
    def __init__(self, cfg: ControllerConfig) -> None:
        super().__init__(cfg)

        # Load MJCF
        spec = mujoco.MjSpec.from_file(self.robot.cfg.mjcf_path)

        # Clear any existing actuators (use programmatic definition)
        spec.actuators.clear()
        
        # Add ground plane (infinite plane at z=0)
        ground_geom = spec.worldbody.add_geom()
        ground_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground_geom.size[:] = [0, 0, 0.05]  # Infinite plane with 0.05m thickness for visuals
        ground_geom.rgba[:] = [0.5, 0.5, 0.5, 1.0]  # Gray color
        ground_geom.friction[:] = [1.0, 0.005, 0.0001]

        # Add actuators based on robot config
        self._add_actuators(spec, self.robot.cfg)
        
        # Compile spec to model
        self.mj_model = spec.compile()
        self.mj_model.opt.timestep = self.cfg.physics_dt
        self.decimation = self.cfg.mujoco.decimation
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.mj_data.qpos = np.concatenate(
            [
                np.array(cfg.mujoco.init_pos, dtype=np.float32),
                np.array(cfg.mujoco.init_quat, dtype=np.float32),
                self.robot.default_joint_pos.numpy(),
            ]
        )
        mujoco.mj_forward(self.mj_model, self.mj_data)

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

    def ctrl_step(self, dof_targets: torch.Tensor):
        """Apply control using manual PD torque computation (matches booster_deploy)."""
        dof_targets = dof_targets.cpu().numpy()

        if self.vel_command is not None:
            self.update_command()

        # Get initial state
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]

        # Get PD gains
        kp = np.asarray(self.robot.cfg.joint_stiffness, dtype=np.float32)
        kd = np.asarray(self.robot.cfg.joint_damping, dtype=np.float32)
        effort = np.asarray(self.robot.cfg.effort_limit, dtype=np.float32)

        # Manual PD control with per-step state updates\
        for i in range(self.decimation):
            # Compute PD torque based on CURRENT state
            tau = kp * (dof_targets - dof_pos) - kd * dof_vel
            tau = np.clip(tau, -effort, effort)

            # Apply torques (not positions!)
            self.mj_data.ctrl[:] = tau

            # Step physics
            mujoco.mj_step(self.mj_model, self.mj_data)

            # CRITICAL: Update state for next substep
            dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6:]

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
                
    def _add_actuators(self, spec: mujoco.MjSpec, robot_cfg: RobotConfig) -> None:
        """Add one position actuator per joint using RobotConfig gains/limits.

        This mirrors the mjlab helper but keeps deploy free of mjlab dependency.
        """

        if len(spec.actuators) > 0:
            return  # Actuators already present in the MJCF

        for i, joint_name in enumerate(robot_cfg.sim_joint_names):
            actuator = spec.add_actuator(name=joint_name, target=joint_name)

            # Configure as motor actuator (direct torque)
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
            
            # Unity gain (passthrough)
            actuator.gainprm[0] = 1.0

            # Torque limits (motor accepts torques in ctrl[])
            effort_limit = float(robot_cfg.effort_limit[i])
            actuator.ctrllimited = True
            actuator.ctrlrange[:] = np.array([-effort_limit, effort_limit], dtype=np.float64)

            # Set joint armature (reflected inertia) - CRITICAL for stability!
            armature = float(robot_cfg.joint_armature[i])
            spec.joint(joint_name).armature = armature

