# ====== Scene Definition ======
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainImporterCfg

from colosseum.robots.cartpole.cartpole_constants import CARTPOLE_ROBOT_CFG

scene_config = SceneCfg(
    terrain=TerrainImporterCfg(
        terrain_type="plane",
    ),
    num_envs=512,
    extent=1.0,
    entities={"robot": CARTPOLE_ROBOT_CFG},
)

# ====== Viewer Configuration ======
from mjlab.viewer import ViewerConfig

viewer_config = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="pole",
    distance=3.0,
    elevation=10.0,
    azimuth=90.0,
)


# ====== Simulation Configuration ======
from mjlab.sim import MujocoCfg, SimulationCfg

simulation_config = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.02,
        iterations=1,
    ),
)
