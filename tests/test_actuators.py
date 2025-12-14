"""Test T1 actuators by holding robot in home keyframe position."""

import mujoco
import mujoco.viewer as viewer
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainImporterCfg

from colosseum.robots.booster_t1.constants import (
    get_t1_fullbody_robot_cfg,
    get_t1_locomotion_robot_cfg,
)


def get_home_positions(use_fullbody: bool) -> dict[str, float]:
    """Get home keyframe joint positions as a flat dictionary.

    Args:
        use_fullbody: Whether using full-body configuration

    Returns:
        Dictionary mapping joint names to positions
    """
    # Expand patterns to explicit joint names for locomotion (12-DOF)
    if not use_fullbody:
        return {
            "Left_Hip_Pitch": -0.2,
            "Right_Hip_Pitch": -0.2,
            "Left_Knee_Pitch": 0.4,
            "Right_Knee_Pitch": 0.4,
            "Left_Ankle_Pitch": -0.2,
            "Right_Ankle_Pitch": -0.2,
            "Left_Hip_Roll": 0.0,
            "Right_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Left_Ankle_Roll": 0.0,
            "Right_Ankle_Roll": 0.0,
        }
    else:
        # Full-body (23-DOF)
        return {
            # Head
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            # Arms
            "Left_Shoulder_Pitch": -0.0,
            "Left_Shoulder_Roll": -1.3,
            "Left_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": -0.4,
            "Right_Shoulder_Pitch": -0.0,
            "Right_Shoulder_Roll": 1.3,
            "Right_Elbow_Pitch": 0.0,
            "Right_Elbow_Yaw": 0.4,
            # Waist
            "Waist": 0.0,
            # Legs
            "Left_Hip_Pitch": -0.2,
            "Right_Hip_Pitch": -0.2,
            "Left_Knee_Pitch": 0.4,
            "Right_Knee_Pitch": 0.4,
            "Left_Ankle_Pitch": -0.2,
            "Right_Ankle_Pitch": -0.2,
            "Left_Hip_Roll": 0.0,
            "Right_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Left_Ankle_Roll": 0.0,
            "Right_Ankle_Roll": 0.0,
        }


def apply_home_positions(
    model: mujoco.MjModel, data: mujoco.MjData, positions: dict[str, float]
):
    """Apply home keyframe positions to actuators.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        positions: Dictionary mapping joint names to target positions
    """
    for joint_name, target_pos in positions.items():
        try:
            # Add robot/ prefix for actuator names in scene
            actuator_name = f"robot/{joint_name}"
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            data.ctrl[actuator_id] = target_pos
        except Exception:
            # Silently skip if actuator doesn't exist (e.g., passive joints in locomotion mode)
            pass


def main():
    print("\n=== Testing T1 Actuators - Home Keyframe Hold ===")

    # Choose robot configuration
    robot_config = input(
        "\nSelect robot configuration (1=12-DOF locomotion, 2=23-DOF full-body, default=1): "
    ).strip()
    use_fullbody = robot_config == "2"

    print(f"\nRobot: {'23-DOF Full-Body' if use_fullbody else '12-DOF Locomotion'}")

    # Get robot config
    robot_cfg = (
        get_t1_fullbody_robot_cfg() if use_fullbody else get_t1_locomotion_robot_cfg()
    )

    # Create scene with terrain using mjlab's Scene
    scene_cfg = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        num_envs=1,
        extent=2.0,
        entities={"robot": robot_cfg},
    )

    print("\nCreating scene with terrain...")
    scene = Scene(scene_cfg, device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)

    # Reset to initial state
    mujoco.mj_resetData(model, data)

    print(f"\nRobot loaded with {model.nq} DOF, {model.nu} actuators")

    # Get home positions
    home_positions = get_home_positions(use_fullbody)

    print("\n=== Home Keyframe Positions ===")
    for joint_name, pos in home_positions.items():
        print(f"{joint_name:<25}: {pos:>7.3f} rad")

    print("\n=== Starting Simulation ===")
    print("The robot will hold its home keyframe position.")
    print("Controls:")
    print("  SPACE: Pause/unpause simulation")
    print("  ESC: Exit")

    # Launch viewer
    with viewer.launch_passive(model, data) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        v.sync()

        paused = False
        sim_time = 0.0

        while v.is_running():
            if not paused:
                # Apply home positions to hold the robot
                apply_home_positions(model, data, home_positions)

                # Step simulation
                mujoco.mj_step(model, data)
                sim_time += model.opt.timestep

                # Print debug info every 0.5 seconds
                if int(sim_time * 2) % 2 == 0 and sim_time > 0:
                    trunk_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, "robot/Trunk"
                    )
                    trunk_height = data.xpos[trunk_id][2]
                    print(
                        f"Time: {sim_time:>6.2f}s | Trunk height: {trunk_height:.3f}m",
                        end="\r",
                    )

            v.sync()

    print("\n\nTest complete!")


if __name__ == "__main__":
    main()
