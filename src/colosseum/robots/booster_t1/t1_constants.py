import mujoco
from mjlab.entity import EntityCfg
from colosseum.utils import src_dir
import mujoco.viewer as viewer
from mjlab.entity import Entity

# Path to the T1 XML file
T1_XML = src_dir() / "robots" / "booster_t1" / "xmls" / "T1_12dof.xml"
assert T1_XML.exists(), f"XML not found: {T1_XML}"

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(T1_XML))
    
    # Debug: print what we loaded
    print(f"Bodies: {len(list(spec.bodies))}")
    print(f"Geoms: {len(list(spec.geoms))}")
    print(f"Joints: {len(list(spec.joints))}")
    
    return spec

T1_ROBOT_CFG = EntityCfg(spec_fn=get_spec)

# Test script to verify the model loads
if __name__ == "__main__":
    
    robot = Entity(T1_ROBOT_CFG)
    viewer.launch(robot.spec.compile())