"""Debug why robot falls when holding default joint positions.

This tests if the default positions are a stable stance by commanding
them without policy control.
"""

import mujoco
import numpy as np
import time

from colosseum.tasks.velocity.deploy.t1_23dof.config import T1_23DOF_VELOCITY_ROUGH

def analyze_default_pose_stability():
    """Check if default pose is stable without policy control."""
    
    # Load config
    cfg = T1_23DOF_VELOCITY_ROUGH
    robot_cfg = cfg.robot
    
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(robot_cfg.mjcf_path)
    data = mujoco.MjData(model)
    
    # Initialize at configured height
    data.qpos[:3] = cfg.mujoco.init_pos  # Set base position
    data.qpos[3:7] = cfg.mujoco.init_quat  # Set base orientation
    
    # Set joints to default positions (in simulation order)
    # Default positions are in real robot order, but for T1 real==sim
    default_pos = np.array(robot_cfg.default_joint_pos)
    data.qpos[7:] = default_pos
    
    print("=" * 80)
    print("DEFAULT POSE STABILITY TEST")
    print("=" * 80)
    print(f"Initial base position: {cfg.mujoco.init_pos}")
    print(f"Initial base height: {cfg.mujoco.init_pos[2]:.4f}m")
    print(f"\nDefault joint positions (rad):")
    for i, (name, pos) in enumerate(zip(robot_cfg.joint_names, default_pos)):
        print(f"  {name:25s}: {pos:7.3f} rad ({np.degrees(pos):6.2f}°)")
    
    # Forward kinematics to get initial state
    mujoco.mj_forward(model, data)
    
    initial_com_z = data.subtree_com[1, 2]  # Body 1 is trunk
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
    
    initial_left_foot_z = data.xpos[left_foot_id, 2]
    initial_right_foot_z = data.xpos[right_foot_id, 2]
    
    print(f"\nInitial state (t=0):")
    print(f"  COM height: {initial_com_z:.4f}m")
    print(f"  Left foot Z: {initial_left_foot_z:.4f}m")
    print(f"  Right foot Z: {initial_right_foot_z:.4f}m")
    print(f"  Feet above ground: {initial_left_foot_z:.4f}m")
    
    # Get actuator mapping
    print(f"\nActuator configuration:")
    print(f"  Number of actuators: {model.nu}")
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        kp = model.actuator_gainprm[i, 0]
        kd = -model.actuator_biasprm[i, 2]  # Damping is negated
        print(f"  Actuator {i:2d} ({joint_name:25s}): Kp={kp:7.2f}, Kd={kd:6.2f}")
    
    # Simulate holding default positions
    print(f"\nSimulating 2 seconds holding default positions...")
    print(f"{'Time(s)':>8} {'Base Z':>8} {'Base dZ/dt':>10} {'COM Z':>8} {'L_Foot Z':>10} {'Contact':>8}")
    print("-" * 70)
    
    dt = model.opt.timestep
    total_time = 2.0
    steps = int(total_time / dt)
    
    max_z_velocity = 0.0
    contact_established = False
    
    for step in range(steps):
        # Command default positions (PD control will try to reach them)
        data.ctrl[:] = default_pos
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Record metrics every 0.1 seconds
        if step % int(0.1 / dt) == 0:
            t = step * dt
            base_z = data.qpos[2]
            base_z_vel = data.qvel[2]
            com_z = data.subtree_com[1, 2]
            left_foot_z = data.xpos[left_foot_id, 2]
            
            # Check contact forces
            contact_force = 0.0
            for i in range(data.ncon):
                contact = data.contact[i]
                geom1 = model.geom_bodyid[contact.geom1]
                geom2 = model.geom_bodyid[contact.geom2]
                if geom1 in [left_foot_id, right_foot_id] or geom2 in [left_foot_id, right_foot_id]:
                    # Get contact force magnitude
                    contact_force += np.linalg.norm(contact.frame[:3])
            
            if contact_force > 100:
                contact_established = True
            
            max_z_velocity = max(max_z_velocity, abs(base_z_vel))
            
            print(f"{t:8.2f} {base_z:8.4f} {base_z_vel:10.4f} {com_z:8.4f} {left_foot_z:10.4f} {contact_force:8.1f}")
    
    # Final analysis
    final_base_z = data.qpos[2]
    final_base_z_vel = data.qvel[2]
    final_com_z = data.subtree_com[1, 2]
    
    print("\n" + "=" * 80)
    print("FINAL STATE ANALYSIS")
    print("=" * 80)
    print(f"Base height change: {initial_com_z:.4f}m → {final_base_z:.4f}m (Δ = {final_base_z - cfg.mujoco.init_pos[2]:.4f}m)")
    print(f"COM height change: {initial_com_z:.4f}m → {final_com_z:.4f}m (Δ = {final_com_z - initial_com_z:.4f}m)")
    print(f"Max Z velocity: {max_z_velocity:.4f} m/s")
    print(f"Final Z velocity: {final_base_z_vel:.4f} m/s")
    print(f"Contact established: {contact_established}")
    
    # Stability criteria
    is_stable = (
        abs(final_base_z_vel) < 0.1 and  # Not falling
        abs(final_base_z - cfg.mujoco.init_pos[2]) < 0.05 and  # Height maintained
        contact_established  # Feet touching ground
    )
    
    print(f"\nSTABILITY: {'✓ STABLE' if is_stable else '✗ UNSTABLE'}")
    
    if not is_stable:
        print("\nPOSSIBLE ISSUES:")
        if not contact_established:
            print("  - Feet not touching ground (initial height too high)")
        if abs(final_base_z_vel) > 0.1:
            print(f"  - Robot falling (Z velocity = {final_base_z_vel:.4f} m/s)")
        if abs(final_base_z - cfg.mujoco.init_pos[2]) > 0.05:
            print(f"  - Height not maintained (dropped {cfg.mujoco.init_pos[2] - final_base_z:.4f}m)")
        print("\nSUGGESTIONS:")
        print(f"  1. Try init_pos height = {initial_left_foot_z + 0.633:.4f}m (leg length from ground contact test)")
        print(f"  2. Check if default_joint_pos matches MuJoCo model's home position")
        print(f"  3. Verify PD gains are strong enough to hold pose against gravity")


if __name__ == "__main__":
    analyze_default_pose_stability()
