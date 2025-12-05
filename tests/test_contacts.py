# src/colosseum/tests/test_t1_contacts.py
import mujoco.viewer as viewer
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainImporterCfg

from colosseum.robots.booster_t1.t1_constants import get_t1_locomotion_robot_cfg

# Define contact sensor
feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_foot_link|right_foot_link)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    debug=True,  # This prints what sensors are created!
)

# Create scene
scene_cfg = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    num_envs=1,
    extent=1.0,
    entities={"robot": get_t1_locomotion_robot_cfg()},
    sensors=(feet_ground_cfg,),
)

if __name__ == "__main__":

    # Create scene
    scene = Scene(scene_cfg, device="cuda:0")
    model = scene.compile()

    # Create simulation
    sim_cfg = SimulationCfg(njmax=64)
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device="cuda")

    # Attach scene to simulation
    scene.initialize(sim.mj_model, sim.model, sim.data)

    print("\n=== Sensor created ===")
    sensor: ContactSensor = scene["feet_ground_contact"]
    print(f"Sensor data shape - found: {sensor.data.found.shape}")
    print(f"Sensor data shape - force: {sensor.data.force.shape}")

    print("\n=== Running simulation ===")
    for i in range(1000):
        # Apply zero control (robot just stands/falls naturally)
        sim.data.ctrl[:, :] = 0.0

        # Step physics
        sim.step()

        # Update sensor readings (IMPORTANT!)
        sensor.update(dt=0.01)

        if i % 20 == 0:
            print(f"\nStep {i}:")
            print(f"  Left foot contact: {sensor.data.found[0, 0].item()}")
            print(f"  Right foot contact: {sensor.data.found[0, 1].item()}")
            print(f"  Left foot force Z: {sensor.data.force[0, 0, 2].item():.2f} N")
            print(f"  Right foot force Z: {sensor.data.force[0, 1, 2].item():.2f} N")

    print("\n=== Launching viewer ===")
    viewer.launch(sim.mj_model)
