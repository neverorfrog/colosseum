#!/usr/bin/env python3
"""Check ground contact and COM stability.

Diagnostic for understanding why robot is falling.
"""

import numpy as np
import mujoco
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from colosseum.robots.t1_23dof.deploy_config import T1_23DOF_ROBOT_CFG
from colosseum.tasks.velocity.deploy.t1_23dof.config import T1_23DOF_VELOCITY_ROUGH
from mjlab.utils.spec import create_position_actuator


def test_ground_contact_and_com():
    """Test if feet are contacting ground and COM is stable."""
    print("=" * 80)
    print("GROUND CONTACT & CENTER OF MASS TEST")
    print("=" * 80)

    cfg = T1_23DOF_VELOCITY_ROUGH
    robot_cfg = cfg.robot

    # Compile model
    spec = mujoco.MjSpec.from_file(robot_cfg.mjcf_path)
    spec.actuators.clear()

    # Add ground plane
    ground_geom = spec.worldbody.add_geom()
    ground_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground_geom.size[:] = [0, 0, 0.05]
    ground_geom.rgba[:] = [0.5, 0.5, 0.5, 1.0]
    ground_geom.friction[:] = [1.0, 0.005, 0.0001]

    # Add actuators
    for i, joint_name in enumerate(robot_cfg.sim_joint_names):
        create_position_actuator(
            spec,
            joint_name,
            stiffness=robot_cfg.joint_stiffness[i],
            damping=robot_cfg.joint_damping[i],
            effort_limit=robot_cfg.effort_limit[i],
        )

    model = spec.compile()
    model.opt.timestep = cfg.physics_dt
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Initialize
    qpos = np.concatenate([
        np.array(cfg.mujoco.init_pos, dtype=np.float32),
        np.array(cfg.mujoco.init_quat, dtype=np.float32),
        np.array(robot_cfg.default_joint_pos, dtype=np.float32),
    ])
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    print(f"\n1. INITIAL STATE")
    print(f"   Base position: {data.qpos[:3]}")
    print(f"   Base quaternion: {data.qpos[3:7]}")

    # Get body IDs for feet
    try:
        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
        print(f"   Left foot ID: {left_foot_id}")
        print(f"   Right foot ID: {right_foot_id}")
    except:
        print(f"   ⚠️  Could not find foot bodies")
        left_foot_id = None
        right_foot_id = None

    # Get COM
    print(f"\n2. CENTER OF MASS")
    print(f"   Total mass: {np.sum(model.body_mass):.2f} kg")
    
    # Compute COM from all bodies
    total_mass = 0
    com = np.zeros(3)
    for i in range(model.nbody):
        mass = model.body_mass[i]
        pos = data.xpos[i]
        com += mass * pos
        total_mass += mass
    com /= total_mass
    
    print(f"   COM position: {com}")
    print(f"   Base (root) position: {data.qpos[:3]}")
    print(f"   COM offset from base: {com - data.qpos[:3]}")

    # Run simulation and check
    print(f"\n3. GROUND CONTACT CHECK")
    
    # Set control targets to default positions
    dof_targets = np.array(robot_cfg.default_joint_pos, dtype=np.float32)

    contact_logs = {
        'foot_z_min': [],
        'foot_z_max': [],
        'total_contact_force': [],
        'base_z_vel': [],
        'com_z': [],
    }

    for step in range(100):
        # Apply control
        dof_pos = data.qpos[7:].astype(np.float32)
        dof_vel = data.qvel[6:].astype(np.float32)
        kp = robot_cfg.joint_stiffness
        kd = robot_cfg.joint_damping

        ctrl = np.clip(
            kp * (dof_targets - dof_pos) - kd * dof_vel,
            model.actuator_forcerange[:, 0],
            model.actuator_forcerange[:, 1],
        )

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % 10 == 0:
            # Get feet positions
            if left_foot_id is not None and right_foot_id is not None:
                left_foot_z = data.xpos[left_foot_id, 2]
                right_foot_z = data.xpos[right_foot_id, 2]
                contact_logs['foot_z_min'].append(min(left_foot_z, right_foot_z))
                contact_logs['foot_z_max'].append(max(left_foot_z, right_foot_z))

            # Total contact force
            contact_force = np.sum(np.abs(data.cfrc_ext[:, :3]))
            contact_logs['total_contact_force'].append(contact_force)

            # Base Z velocity
            base_z_vel = data.qvel[2]
            contact_logs['base_z_vel'].append(base_z_vel)

            # COM Z position
            com = np.zeros(3)
            total_mass = 0
            for i in range(model.nbody):
                mass = model.body_mass[i]
                pos = data.xpos[i]
                com += mass * pos
                total_mass += mass
            com /= total_mass
            contact_logs['com_z'].append(com[2])

    # Analyze results
    if contact_logs['foot_z_min']:
        contact_logs['foot_z_min'] = np.array(contact_logs['foot_z_min'])
        contact_logs['foot_z_max'] = np.array(contact_logs['foot_z_max'])
        contact_logs['total_contact_force'] = np.array(contact_logs['total_contact_force'])
        contact_logs['base_z_vel'] = np.array(contact_logs['base_z_vel'])
        contact_logs['com_z'] = np.array(contact_logs['com_z'])

        print(f"\n   After 100 steps:")
        print(f"   Foot Z range: min={contact_logs['foot_z_min'][-1]:.4f}, max={contact_logs['foot_z_max'][-1]:.4f}")
        print(f"   Foot contact difference: {contact_logs['foot_z_max'][-1] - contact_logs['foot_z_min'][-1]:.4f}m")
        
        if contact_logs['foot_z_min'][-1] < 0.01:
            print(f"   ✓ Feet are touching ground (z < 0.01m)")
        else:
            print(f"   ❌ Feet NOT touching ground! (z = {contact_logs['foot_z_min'][-1]:.4f}m)")

        print(f"\n   Contact forces (should increase as feet settle):")
        print(f"   Start: {contact_logs['total_contact_force'][0]:.2f} N")
        print(f"   End:   {contact_logs['total_contact_force'][-1]:.2f} N")

        print(f"\n   Base Z velocity (should approach 0):")
        print(f"   Start: {contact_logs['base_z_vel'][0]:.4f} m/s")
        print(f"   End:   {contact_logs['base_z_vel'][-1]:.4f} m/s")
        print(f"   Min:   {np.min(contact_logs['base_z_vel']):.4f} m/s")

        print(f"\n   COM Z position (should stabilize):")
        print(f"   Start: {contact_logs['com_z'][0]:.4f} m")
        print(f"   End:   {contact_logs['com_z'][-1]:.4f} m")
        print(f"   Drift: {contact_logs['com_z'][-1] - contact_logs['com_z'][0]:.4f} m")

        # Diagnosis
        print(f"\n4. DIAGNOSIS")
        if contact_logs['base_z_vel'][-1] < -0.1:
            print(f"   ❌ Robot still falling! (z_vel = {contact_logs['base_z_vel'][-1]:.4f})")
            if contact_logs['foot_z_min'][-1] < 0.01:
                print(f"      Feet touching, but control can't stabilize falling")
                print(f"      → Check if ground friction is too low")
                print(f"      → Check if PD gains are correct for stability")
            else:
                print(f"      Feet NOT contacting ground!")
                print(f"      → Check ground plane collision geometry")
                print(f"      → Check foot body geometry")
                print(f"      → Check init height (currently {cfg.mujoco.init_pos[2]}m)")
        else:
            print(f"   ✓ Robot falling motion stabilized")


if __name__ == "__main__":
    test_ground_contact_and_com()
