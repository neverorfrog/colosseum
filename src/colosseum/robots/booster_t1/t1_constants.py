import mujoco
from mjlab.entity import EntityCfg, Entity
from colosseum.utils import src_dir
import mujoco.viewer as viewer

# Paths
T1_LOCOMOTION_XML = src_dir() / "robots" / "booster_t1" / "xmls" / "T1_locomotion.xml"
T1_SERIAL_XML = src_dir() / "robots" / "booster_t1" / "xmls" / "T1_serial.xml"

assert T1_LOCOMOTION_XML.exists(), f"XML not found: {T1_LOCOMOTION_XML}"
assert T1_SERIAL_XML.exists(), f"XML not found: {T1_SERIAL_XML}"

# Action scales
T1_LOCOMOTION_ACTION_SCALE = 0.5  # For training
T1_SERIAL_ACTION_SCALE = 0.5      # For deployment

def get_t1_locomotion_spec() -> mujoco.MjSpec:
    """T1 with legs only (12 DOF) - for training."""
    return mujoco.MjSpec.from_file(str(T1_LOCOMOTION_XML))

def get_t1_serial_spec() -> mujoco.MjSpec:
    """T1 full body (23 DOF) - for deployment."""
    return mujoco.MjSpec.from_file(str(T1_SERIAL_XML))

# Locomotion keyframe - simple, upright pose for training
LOCOMOTION_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.67),
    joint_pos={
        ".*Hip_Pitch": -0.2,
        ".*Knee_Pitch": 0.4,
        ".*Ankle_Pitch": -0.25,
        ".*": 0.0,  # All other joints (Hip_Roll, Hip_Yaw, Ankle_Roll)
    },
    joint_vel={".*": 0.0},
)

def get_t1_locomotion_robot_cfg() -> EntityCfg:
    """Get T1 locomotion config (12 DOF legs only) - for training."""
    return EntityCfg(
        spec_fn=get_t1_locomotion_spec,
        init_state=LOCOMOTION_HOME_KEYFRAME,
    )

# Convenience shorthand
T1_ROBOT_CFG = get_t1_locomotion_robot_cfg()

if __name__ == "__main__":
    print("\n=== Testing T1 Locomotion Model ===")
    robot = Entity(T1_ROBOT_CFG)
    model = robot.spec.compile()
    data = mujoco.MjData(model)

    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)

    # Compare keyframe vs actual
    print("\n=== Keyframe vs Actual Joint Positions ===")
    keyframe = model.key("init_state")
    print(f"{'Joint Name':20s} | {'Keyframe (rad)':>15s} | {'Actual (rad)':>15s} | {'Diff (rad)':>15s}")
    print("-" * 75)
    for i, name in enumerate(robot.joint_names):
        keyframe_pos = keyframe.qpos[i]
        actual_pos = data.qpos[i]
        diff = actual_pos - keyframe_pos
        print(f"{name:20s} | {keyframe_pos:15.6f} | {actual_pos:15.6f} | {diff:15.6f}")

    print(f"\n=== Total DOF: {len(robot.joint_names)} ===")
    print("Launching viewer...")
    print("TIP: Press SPACE to pause/unpause the simulation")

    # Create viewer handle with paused state
    with viewer.launch_passive(model, data) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        v.sync()

        # Start paused - user must press space to start
        paused = True
        while v.is_running():
            if not paused:
                mujoco.mj_step(model, data)
            v.sync()