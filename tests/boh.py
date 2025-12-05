import mujoco

xml_path = "src/colosseum/robots/booster_t1/xmls/T1_23dof.xml"
spec = mujoco.MjSpec.from_file(xml_path)
model = spec.compile()

print(f"Total joints: {model.njnt}")
print(f"Total DOF (nv): {model.nv}")
print(f"Total actuators: {model.nu}")

print("\n=== Actuated Joints ===")
for i in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    joint_id = model.actuator_trnid[i, 0]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    print(f"{i}: {actuator_name} -> {joint_name}")