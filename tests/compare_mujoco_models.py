"""Compare MuJoCo model setup between training (play.py) and deployment."""

import mujoco
from colosseum.robots.t1_23dof.deploy_config import T1_23DOF_ROBOT_CFG
from colosseum.robots.t1_23dof.constants import get_robot_cfg
from mjlab.entity import Entity
from mjlab.utils.spec import create_position_actuator

print("=" * 80)
print("TRAINING SETUP (play.py / mjlab)")
print("=" * 80)

# Training: Load robot config and create entity
train_cfg = get_robot_cfg()
train_entity = Entity(train_cfg)
train_model = train_entity.spec.compile()

print(f"\nModel properties:")
print(f"  Timestep: {train_model.opt.timestep}")
print(f"  Integrator: {train_model.opt.integrator}")
print(f"  Iterations: {train_model.opt.iterations}")
print(f"  LS iterations: {train_model.opt.ls_iterations}")
print(f"  Num actuators: {train_model.nu}")

print(f"\nFirst 5 joints:")
for i in range(5):
    jnt_name = train_model.joint(i+1).name
    qpos_adr = train_model.jnt_qposadr[i+1] - 7
    print(f"  {jnt_name:25s}: arm={train_model.dof_armature[qpos_adr]:.6f}, damp={train_model.dof_damping[qpos_adr]:.6f}, fric={train_model.dof_frictionloss[qpos_adr]:.6f}")

print(f"\nFirst 5 actuators:")
for i in range(min(5, train_model.nu)):
    act_name = train_model.actuator(i).name
    print(f"  {act_name:25s}: gain={train_model.actuator_gainprm[i,0]:.2f}, bias1={train_model.actuator_biasprm[i,1]:.2f}, bias2={train_model.actuator_biasprm[i,2]:.2f}")

print("\n" + "=" * 80)
print("DEPLOYMENT SETUP (MujocoController)")
print("=" * 80)

# Deployment: Replicate what MujocoController does
spec = mujoco.MjSpec.from_file(str(T1_23DOF_ROBOT_CFG.mjcf_path))
spec.actuators.clear()

# Add actuators like deployment does
for i, joint_name in enumerate(T1_23DOF_ROBOT_CFG.sim_joint_names):
    create_position_actuator(
        spec,
        joint_name,
        stiffness=T1_23DOF_ROBOT_CFG.joint_stiffness[i],
        damping=T1_23DOF_ROBOT_CFG.joint_damping[i],
        effort_limit=T1_23DOF_ROBOT_CFG.effort_limit[i],
    )

deploy_model = spec.compile()

print(f"\nModel properties:")
print(f"  Timestep: {deploy_model.opt.timestep}")
print(f"  Integrator: {deploy_model.opt.integrator}")
print(f"  Iterations: {deploy_model.opt.iterations}")
print(f"  LS iterations: {deploy_model.opt.ls_iterations}")
print(f"  Num actuators: {deploy_model.nu}")

print(f"\nFirst 5 joints:")
for i in range(5):
    jnt_name = deploy_model.joint(i+1).name
    qpos_adr = deploy_model.jnt_qposadr[i+1] - 7
    print(f"  {jnt_name:25s}: arm={deploy_model.dof_armature[qpos_adr]:.6f}, damp={deploy_model.dof_damping[qpos_adr]:.6f}, fric={deploy_model.dof_frictionloss[qpos_adr]:.6f}")

print(f"\nFirst 5 actuators:")
for i in range(min(5, deploy_model.nu)):
    act_name = deploy_model.actuator(i).name
    print(f"  {act_name:25s}: gain={deploy_model.actuator_gainprm[i,0]:.2f}, bias1={deploy_model.actuator_biasprm[i,1]:.2f}, bias2={deploy_model.actuator_biasprm[i,2]:.2f}")

print("\n" + "=" * 80)
print("DIFFERENCES")
print("=" * 80)

print("\nTimestep: {} vs {}".format(train_model.opt.timestep, deploy_model.opt.timestep))
print("Num actuators: {} vs {}".format(train_model.nu, deploy_model.nu))

if train_model.nu == deploy_model.nu:
    print("\nActuator gain differences:")
    for i in range(train_model.nu):
        train_gain = train_model.actuator_gainprm[i, 0]
        deploy_gain = deploy_model.actuator_gainprm[i, 0]
        if abs(train_gain - deploy_gain) > 0.01:
            print(f"  {train_model.actuator(i).name:25s}: {train_gain:.2f} vs {deploy_gain:.2f}")
