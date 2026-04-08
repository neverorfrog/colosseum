"""Booster T1 head-mounted RGB-D camera demo.

This demo loads the Booster T1 MJCF from the sibling colosseum repository,
mounts an ``mjlab`` ``CameraSensor`` on the robot head, and visualizes both RGB
and depth outputs.

Run with:
  uv run python scripts/demos/booster_t1_rgbd_camera.py --viewer viser

If your colosseum checkout is not in the default sibling path, pass the XML path:
  uv run python scripts/demos/booster_t1_rgbd_camera.py \
    --t1-xml /abs/path/to/colosseum/src/colosseum/robots/t1_23dof/xmls/T1_23dof.xml \
    --viewer viser

In Viser:
  - open the printed URL (usually http://localhost:8080)
  - open the "Camera Feeds" tab
  - inspect both RGB and depth panels for sensor ``head_rgbd``
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# Joint home configuration used by colosseum's T1 setup.
HOME_QPOS: dict[str, float] = {
  "AAHead_yaw": 0.0,
  "Head_pitch": 0.0,
  "Left_Shoulder_Pitch": 0.2,
  "Left_Shoulder_Roll": -1.3,
  "Left_Elbow_Pitch": 0.0,
  "Left_Elbow_Yaw": -0.5,
  "Right_Shoulder_Pitch": 0.2,
  "Right_Shoulder_Roll": 1.3,
  "Right_Elbow_Pitch": 0.0,
  "Right_Elbow_Yaw": 0.5,
  "Waist": 0.0,
  "Left_Hip_Pitch": -0.2,
  "Left_Hip_Roll": 0.0,
  "Left_Hip_Yaw": 0.0,
  "Left_Knee_Pitch": 0.4,
  "Left_Ankle_Pitch": -0.2,
  "Left_Ankle_Roll": 0.0,
  "Right_Hip_Pitch": -0.2,
  "Right_Hip_Roll": 0.0,
  "Right_Hip_Yaw": 0.0,
  "Right_Knee_Pitch": 0.4,
  "Right_Ankle_Pitch": -0.2,
  "Right_Ankle_Roll": 0.0,
}

STANDING_BASE_POS = (0.0, 0.0, 0.665)
STANDING_BASE_QUAT = (1.0, 0.0, 0.0, 0.0)

# RealSense-like calibrated intrinsics.
REAL_CAM_NAME = "d455_color"
REAL_CAM_WIDTH = 1280
REAL_CAM_HEIGHT = 720
REAL_CAM_K = np.array(
  [
    [646.0612, 0.0, 644.3064],
    [0.0, 645.1986, 357.1254],
    [0.0, 0.0, 1.0],
  ],
  dtype=np.float64,
)


def _fovy_from_intrinsics(height: int, fy: float) -> float:
  return float(np.degrees(2.0 * np.arctan(height / (2.0 * fy))))


def _set_mujoco_camera_intrinsics(
  cam: mujoco.MjsCamera,
  width: int,
  height: int,
  fx: float,
  fy: float,
  cx: float,
  cy: float,
) -> None:
  cam.resolution[:] = (width, height)
  # Exact intrinsics when available.
  if hasattr(cam, "focalpixel") and hasattr(cam, "principalpixel"):
    setattr(cam, "focalpixel", np.array([fx, fy], dtype=np.float64))
    setattr(cam, "principalpixel", np.array([cx, cy], dtype=np.float64))
  else:
    # Fallback to FOV-only if this MuJoCo build lacks focalpixel support.
    cam.fovy = _fovy_from_intrinsics(height=height, fy=fy)


class BoosterT1RgbdEnv(ManagerBasedRlEnv):
  """ManagerBasedRlEnv with debug frame overlays for axis visualization."""

  def __init__(self, cfg: ManagerBasedRlEnvCfg, device: str):
    super().__init__(cfg=cfg, device=device)
    self._body_axis_body_id = self.sim.mj_model.body("booster/Trunk").id
    self._camera_axis_cam_id = self.scene["head_rgbd"].camera_idx

  def update_visualizers(self, visualizer) -> None:
    # Keep default manager/sensor visualizations.
    super().update_visualizers(visualizer)

    world_R = np.eye(3, dtype=np.float32)
    for env_idx in visualizer.get_env_indices(self.num_envs):
      # World axis at origin.
      visualizer.add_frame(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotation_matrix=world_R,
        scale=0.30,
        axis_radius=0.008,
        label="world_axis",
      )

      # Robot body axis (Trunk frame).
      body_pos = self.sim.data.xpos[env_idx, self._body_axis_body_id].cpu().numpy()
      body_R = (
        self.sim.data.xmat[env_idx, self._body_axis_body_id].cpu().numpy().reshape(3, 3)
      )
      visualizer.add_frame(
        position=body_pos,
        rotation_matrix=body_R,
        scale=0.22,
        axis_radius=0.007,
        label="booster_body_axis",
      )

      # Camera axis from rendered camera pose.
      cam_pos = self.sim.data.cam_xpos[env_idx, self._camera_axis_cam_id].cpu().numpy()
      cam_R = (
        self.sim.data.cam_xmat[env_idx, self._camera_axis_cam_id]
        .cpu()
        .numpy()
        .reshape(3, 3)
      )
      visualizer.add_frame(
        position=cam_pos,
        rotation_matrix=cam_R,
        scale=0.18,
        axis_radius=0.006,
        label="camera_axis",
      )


def _default_t1_xml_path() -> Path:
  # scripts/demos -> scripts -> mjlab -> SPQR (workspace parent)
  spqr_root = Path(__file__).resolve().parents[3]
  booster_xml = (
    spqr_root
    / "colosseum"
    / "src"
    / "colosseum"
    / "robots"
    / "t1_23dof"
    / "xmls"
    / "T1_23dof_booster.xml"
  )
  if booster_xml.exists() and _xml_has_all_mesh_files(booster_xml):
    return booster_xml

  return (
    spqr_root
    / "colosseum"
    / "src"
    / "colosseum"
    / "robots"
    / "t1_23dof"
    / "xmls"
    / "T1_23dof.xml"
  )


def _xml_has_all_mesh_files(xml_path: Path) -> bool:
  """Return True when all <mesh file=...> entries resolve on disk.

  Some T1 XML variants may reference optional meshes (e.g., Logo.STL) that are
  missing in a local checkout. In that case we fallback to another XML.
  """
  try:
    root = ET.parse(xml_path).getroot()
  except Exception:
    return False

  compiler = root.find("compiler")
  meshdir = ""
  if compiler is not None:
    meshdir = compiler.attrib.get("meshdir", "")

  xml_dir = xml_path.parent
  mesh_base = (xml_dir / meshdir).resolve() if meshdir else xml_dir.resolve()

  for mesh in root.findall("./asset/mesh"):
    mesh_file = mesh.attrib.get("file")
    if mesh_file is None:
      continue
    if not (mesh_base / mesh_file).exists():
      return False
  return True


def _resolve_t1_xml_path(t1_xml: str | None) -> Path:
  if t1_xml is not None:
    xml_path = Path(t1_xml).expanduser().resolve()
  else:
    xml_path = _default_t1_xml_path()

  if not xml_path.exists():
    raise FileNotFoundError(
      "Booster T1 XML not found. Pass --t1-xml with an absolute path. "
      f"Tried: {xml_path}"
    )
  return xml_path


def _get_t1_spec(xml_path: Path) -> mujoco.MjSpec:
  # The XML references meshes via <compiler meshdir="meshes">; loading from file
  # keeps those paths resolved relative to the XML folder.
  spec = mujoco.MjSpec.from_file(str(xml_path))

  # Add calibrated camera to the H2 head link.
  h2 = spec.body("H2")
  cam = h2.add_camera(
    name=REAL_CAM_NAME,
    pos=(0.074, 0.0, 0.11),
    quat=(0.5, 0.5, -0.5, -0.5),
    fovy=_fovy_from_intrinsics(height=REAL_CAM_HEIGHT, fy=float(REAL_CAM_K[1, 1])),
    resolution=[REAL_CAM_WIDTH, REAL_CAM_HEIGHT],
    proj=mujoco.mjtProjection.mjPROJ_PERSPECTIVE,
  )
  _set_mujoco_camera_intrinsics(
    cam=cam,
    width=REAL_CAM_WIDTH,
    height=REAL_CAM_HEIGHT,
    fx=float(REAL_CAM_K[0, 0]),
    fy=float(REAL_CAM_K[1, 1]),
    cx=float(REAL_CAM_K[0, 2]),
    cy=float(REAL_CAM_K[1, 2]),
  )
  return spec


def _get_world_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_string(
    """
    <mujoco>
      <worldbody>
        <light directional="true" diffuse="0.6 0.6 0.6" pos="0 0 5" dir="0 0 -1"/>
        <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" rgba="0.2 0.2 0.2 1"/>
        <geom name="box_1" type="box" size="0.2 0.2 0.2" pos="2.0 0.0 0.2" rgba="1 0 0 1"/>
        <geom name="box_2" type="box" size="0.25 0.15 0.4" pos="2.5 0.8 0.4" rgba="0 1 0 1"/>
        <geom name="cyl_1" type="cylinder" size="0.15 0.4" pos="2.5 -0.8 0.4" rgba="0 0 1 1"/>
      </worldbody>
    </mujoco>
    """
  )


def create_env_cfg(
  t1_xml_path: Path,
  width: int,
  height: int,
  gravity_zero: bool,
) -> ManagerBasedRlEnvCfg:
  robot_cfg = EntityCfg()
  robot_cfg.spec_fn = lambda: _get_t1_spec(t1_xml_path)
  robot_cfg.init_state = EntityCfg.InitialStateCfg(
    pos=STANDING_BASE_POS,
    joint_pos=HOME_QPOS,
    joint_vel={".*": 0.0},
  )

  world_cfg = EntityCfg()
  world_cfg.spec_fn = _get_world_spec

  # Wrap calibrated camera defined in _get_t1_spec().
  rgbd_sensor_cfg = CameraSensorCfg(
    name="head_rgbd",
    camera_name=f"booster/{REAL_CAM_NAME}",
    width=width,
    height=height,
    data_types=("rgb", "depth"),
    use_textures=True,
    use_shadows=False,
  )

  sim_cfg = SimulationCfg()
  if gravity_zero:
    sim_cfg.mujoco.gravity = (0.0, 0.0, 0.0)

  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    sim=sim_cfg,
    scene=SceneCfg(
      num_envs=1,
      env_spacing=0.0,
      extent=4.0,
      entities={"booster": robot_cfg, "world": world_cfg},
      sensors=(rgbd_sensor_cfg,),
    ),
  )

  cfg.viewer.body_name = "booster/H2"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -10.0
  cfg.viewer.azimuth = 100.0

  return cfg


def main(
  viewer: str,
  t1_xml: str | None,
  width: int,
  height: int,
  gravity_zero: bool,
) -> None:
  configure_torch_backends()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  xml_path = _resolve_t1_xml_path(t1_xml)

  env_cfg = create_env_cfg(
    t1_xml_path=xml_path,
    width=width,
    height=height,
    gravity_zero=gravity_zero,
  )
  env = BoosterT1RgbdEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = viewer

  head_yaw_jid = env.unwrapped.sim.mj_model.joint("booster/AAHead_yaw").id
  head_pitch_jid = env.unwrapped.sim.mj_model.joint("booster/Head_pitch").id
  head_yaw_qpos_adr = int(env.unwrapped.sim.mj_model.jnt_qposadr[head_yaw_jid])
  head_pitch_qpos_adr = int(env.unwrapped.sim.mj_model.jnt_qposadr[head_pitch_jid])
  head_yaw_qvel_adr = int(env.unwrapped.sim.mj_model.jnt_dofadr[head_yaw_jid])
  head_pitch_qvel_adr = int(env.unwrapped.sim.mj_model.jnt_dofadr[head_pitch_jid])

  base_jid = env.unwrapped.sim.mj_model.joint("booster/floating_base_joint").id
  base_qpos_adr = int(env.unwrapped.sim.mj_model.jnt_qposadr[base_jid])
  base_qvel_adr = int(env.unwrapped.sim.mj_model.jnt_dofadr[base_jid])

  stance_joint_addrs: list[tuple[int, int, float]] = []
  for joint_name, joint_pos in HOME_QPOS.items():
    if joint_name in {"AAHead_yaw", "Head_pitch"}:
      continue
    jid = env.unwrapped.sim.mj_model.joint(f"booster/{joint_name}").id
    qpos_adr = int(env.unwrapped.sim.mj_model.jnt_qposadr[jid])
    qvel_adr = int(env.unwrapped.sim.mj_model.jnt_dofadr[jid])
    stance_joint_addrs.append((qpos_adr, qvel_adr, joint_pos))

  print("=" * 72)
  print("Booster T1 Head RGB-D Camera Demo")
  print(f"  XML: {xml_path}")
  print(f"  Device: {device}")
  print(f"  Viewer: {resolved_viewer}")
  print(f"  Sensor resolution: {width}x{height}")
  print(
    "  Camera intrinsics K: "
    f"fx={REAL_CAM_K[0, 0]:.4f}, fy={REAL_CAM_K[1, 1]:.4f}, "
    f"cx={REAL_CAM_K[0, 2]:.4f}, cy={REAL_CAM_K[1, 2]:.4f}"
  )
  print(
    "  FOV from K: "
    f"fovx={np.degrees(2.0 * np.arctan(width / (2.0 * REAL_CAM_K[0, 0]))):.3f} deg, "
    f"fovy={np.degrees(2.0 * np.arctan(height / (2.0 * REAL_CAM_K[1, 1]))):.3f} deg"
  )
  print("  Stance lock: enabled (robot kept standing while head scans)")
  print("  Debug axes: world + booster body + camera")
  if gravity_zero:
    print("  Gravity: disabled")
  print("  Camera sensor name: head_rgbd")
  if resolved_viewer == "viser":
    print("  In Viser, open 'Camera Feeds' to see RGB + depth streams.")
  print("=" * 72)

  class HeadScanPolicy:
    def __init__(self):
      self.step_count = 0

    def __call__(self, obs) -> torch.Tensor:
      del obs
      t = self.step_count * 0.005
      yaw = 0.6 * np.sin(2.0 * np.pi * 0.12 * t)
      pitch = -0.1 + 0.2 * np.sin(2.0 * np.pi * 0.2 * t)

      sim_data = env.unwrapped.sim.data

      # Keep base fixed at standing pose.
      sim_data.qpos[0, base_qpos_adr : base_qpos_adr + 3] = torch.tensor(
        STANDING_BASE_POS, device=device, dtype=torch.float32
      )
      sim_data.qpos[0, base_qpos_adr + 3 : base_qpos_adr + 7] = torch.tensor(
        STANDING_BASE_QUAT, device=device, dtype=torch.float32
      )
      sim_data.qvel[0, base_qvel_adr : base_qvel_adr + 6] = 0.0

      # Keep non-head joints in home stance.
      for qpos_adr, qvel_adr, joint_pos in stance_joint_addrs:
        sim_data.qpos[0, qpos_adr] = joint_pos
        sim_data.qvel[0, qvel_adr] = 0.0

      # Animate only head joints for scanning.
      sim_data.qpos[0, head_yaw_qpos_adr] = yaw
      sim_data.qpos[0, head_pitch_qpos_adr] = pitch
      sim_data.qvel[0, head_yaw_qvel_adr] = 0.0
      sim_data.qvel[0, head_pitch_qvel_adr] = 0.0

      self.step_count += 1
      return torch.zeros(env.unwrapped.action_space.shape, device=device)

  policy = HeadScanPolicy()

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {viewer}")

  env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Booster T1 RGB-D camera demo")
  parser.add_argument(
    "--viewer",
    default="auto",
    choices=("auto", "native", "viser"),
    help="Viewer backend.",
  )
  parser.add_argument(
    "--t1-xml",
    default=None,
    help="Absolute path to Booster T1 XML (T1_23dof.xml).",
  )
  parser.add_argument(
    "--width", type=int, default=REAL_CAM_WIDTH, help="Camera width."
  )
  parser.add_argument(
    "--height", type=int, default=REAL_CAM_HEIGHT, help="Camera height."
  )
  parser.add_argument(
    "--gravity-zero",
    action="store_true",
    default=False,
    help="Disable gravity to keep the robot still during visualization.",
  )

  args = parser.parse_args()
  main(
    viewer=args.viewer,
    t1_xml=args.t1_xml,
    width=args.width,
    height=args.height,
    gravity_zero=args.gravity_zero,
  )
