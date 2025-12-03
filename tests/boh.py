import mujoco

xml_path = "src/colosseum/robots/booster_t1/xmls/T1_locomotion.xml"  # Fill in your path
spec = mujoco.MjSpec.from_file(xml_path)
model = spec.compile()

print("=== T1 Joint Names ===")
for i in range(model.njnt):
    if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(joint_name)
        
print("=== T1 Geom Names ===")
for i in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    body_id = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if geom_name:
        print(f"Geom {i}: {geom_name} (body: {body_name})")
    else:
        print(f"Geom {i}: <unnamed> (body: {body_name})")