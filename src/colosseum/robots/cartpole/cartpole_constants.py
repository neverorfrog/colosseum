from pathlib import Path
import mujoco
 
from colosseum.utils import src_dir
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import Actuator, ActuatorCfg

CARTPOLE_XML: Path = (
  src_dir() / "robots" / "cartpole" / "xmls" / "cartpole.xml"
)
assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CARTPOLE_XML))

def get_cartpole_robot_cfg() -> EntityCfg:
  """Get a fresh CartPole robot configuration instance."""
  return EntityCfg(spec_fn=get_spec)

if __name__ == "__main__":
  import mujoco.viewer as viewer
  robot = Entity(get_cartpole_robot_cfg())
  viewer.launch(robot.spec.compile())