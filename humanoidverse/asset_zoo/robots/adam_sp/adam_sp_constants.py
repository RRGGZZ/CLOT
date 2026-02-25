"""Adam SP constants for mjlab (actuation and control layer)."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ADAM_SP_XML = Path(__file__).parent / "adam_sp.xml"
assert ADAM_SP_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  # meshdir 是从 XML 中读取的相对路径（如 "meshes_stl_0.25/"），去掉末尾斜杠后作为目录名
  meshdir_clean = meshdir.rstrip("/")
  update_assets(assets, ADAM_SP_XML.parent / meshdir_clean, meshdir_clean)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ADAM_SP_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Motor specs (from adam_sp.py).
##

ARMATURE_130_92_7_P = 0.13426
ARMATURE_50_14A_50_S = 0.1578807
ARMATURE_30_14A_50_S = 0.0423963
ARMATURE_60_17_50_S = 0.23409
ARMATURE_80_20_30_S = 0.281573
ARMATURE_50_52_30_P = 0.0549

NATURAL_FREQ = 5 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_130_92_7_P = ARMATURE_130_92_7_P * NATURAL_FREQ**2
STIFFNESS_50_14A_50_S = ARMATURE_50_14A_50_S * NATURAL_FREQ**2
STIFFNESS_30_14A_50_S = ARMATURE_30_14A_50_S * NATURAL_FREQ**2
STIFFNESS_60_17_50_S = ARMATURE_60_17_50_S * NATURAL_FREQ**2
STIFFNESS_80_20_30_S = ARMATURE_80_20_30_S * NATURAL_FREQ**2
STIFFNESS_50_52_30_P = ARMATURE_50_52_30_P * NATURAL_FREQ**2

DAMPING_130_92_7_P = 2.0 * DAMPING_RATIO * ARMATURE_130_92_7_P * NATURAL_FREQ
DAMPING_50_14A_50_S = 2.0 * DAMPING_RATIO * ARMATURE_50_14A_50_S * NATURAL_FREQ
DAMPING_30_14A_50_S = 2.0 * DAMPING_RATIO * ARMATURE_30_14A_50_S * NATURAL_FREQ
DAMPING_60_17_50_S = 2.0 * DAMPING_RATIO * ARMATURE_60_17_50_S * NATURAL_FREQ
DAMPING_80_20_30_S = 2.0 * DAMPING_RATIO * ARMATURE_80_20_30_S * NATURAL_FREQ
DAMPING_50_52_30_P = 2.0 * DAMPING_RATIO * ARMATURE_50_52_30_P * NATURAL_FREQ

##
# Actuators.
##

ADAM_SP_ACT_LEG_PITCH = BuiltinPositionActuatorCfg(
  joint_names_expr=("hipPitch_.*", "kneePitch_.*"),
  stiffness=305.0,
  damping=5.0,
  effort_limit=230.0,
  armature=0.03,
)

ADAM_SP_ACT_LEG_ROLL = BuiltinPositionActuatorCfg(
  joint_names_expr=("hipRoll_.*",),
  stiffness=255.0,
  damping=3.5,
  effort_limit=160.0,
  armature=0.03,
)

ADAM_SP_ACT_LEG_YAW = BuiltinPositionActuatorCfg(
  joint_names_expr=("hipYaw_.*",),
  stiffness=255.0,
  damping=3.5,
  effort_limit=105.0,
  armature=0.03,
)

ADAM_SP_ACT_FEET_PITCH = BuiltinPositionActuatorCfg(
  joint_names_expr=("anklePitch_.*",),
  stiffness=50.0,
  damping=0.8,
  effort_limit=40.0,
  armature=0.03,
)

ADAM_SP_ACT_FEET_ROLL = BuiltinPositionActuatorCfg(
  joint_names_expr=("ankleRoll_.*",),
  stiffness=30.0,
  damping=0.35,
  effort_limit=12.0,
  armature=0.03,
)

ADAM_SP_ACT_WAIST = BuiltinPositionActuatorCfg(
  joint_names_expr=("waistRoll",  "waistYaw"),
  stiffness=255.0,
  damping=3.5,
  effort_limit=110.0,
  armature=0.03,
)

ADAM_SP_ACT_WAIST_PITCH = BuiltinPositionActuatorCfg(
  joint_names_expr=("waistPitch", ),
  stiffness=305.0,
  damping=5.0,
  effort_limit=110.0,
  armature=0.03,
)

ADAM_SP_ACT_SHOULDER = BuiltinPositionActuatorCfg(
  joint_names_expr=("shoulderPitch_.*", "shoulderRoll_.*", "shoulderYaw_.*"),
  stiffness=40.0,
  damping=1.0,
  effort_limit=65.0,
  armature=0.03,
)

ADAM_SP_ACT_ELBOW = BuiltinPositionActuatorCfg(
  joint_names_expr=("elbow_.*", ),
  stiffness=40.0,
  damping=1.0,
  effort_limit=30.0,
  armature=0.03,
)

ADAM_SP_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ADAM_SP_ACT_LEG_PITCH,
    ADAM_SP_ACT_LEG_ROLL,
    ADAM_SP_ACT_LEG_YAW,
    ADAM_SP_ACT_FEET_PITCH,
    ADAM_SP_ACT_FEET_ROLL,
    ADAM_SP_ACT_WAIST,
    ADAM_SP_ACT_WAIST_PITCH,
    ADAM_SP_ACT_SHOULDER,
    ADAM_SP_ACT_ELBOW,
  ),
  soft_joint_pos_limit_factor=0.9,
)

##
# Initial state and collisions.
##

ADAM_SP_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.90),
  rot=(0.0, 0.0, 0.0, 1.0),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
  joint_pos={
    "hipPitch_Left": -0.32,
    "hipRoll_Left": 0.0,
    "hipYaw_Left": -0.18,
    "kneePitch_Left": 0.66,
    "anklePitch_Left": -0.39,
    "ankleRoll_Left": -0.0,
    "hipPitch_Right": -0.32,
    "hipRoll_Right": -0.0,
    "hipYaw_Right": 0.18,
    "kneePitch_Right": 0.66,
    "anklePitch_Right": -0.39,
    "ankleRoll_Right": -0.0,
    "waistRoll": 0.0,
    "waistPitch": 0.0,
    "waistYaw": 0.0,
    "shoulderPitch_Left": 0.0,
    "shoulderRoll_Left": 0.1,
    "shoulderYaw_Left": 0.0,
    "elbow_Left": -0.3,
    "shoulderPitch_Right": 0.0,
    "shoulderRoll_Right": -0.1,
    "shoulderYaw_Right": 0.0,
    "elbow_Right": -0.3,
  },
  joint_vel={".*": 0.0},
)

ADAM_SP_COLLISIONS = (
  CollisionCfg(geom_names_expr=(r".*",), 
  disable_other_geoms=False),
)


def get_adam_sp_robot_cfg() -> EntityCfg:
  """Build a fresh Adam SP robot configuration."""
  return EntityCfg(
    init_state=ADAM_SP_INIT_STATE,
    collisions=ADAM_SP_COLLISIONS,
    spec_fn=get_spec,
    articulation=ADAM_SP_ARTICULATION,
  )


ADAM_SP_ACTION_SCALE: dict[str, float] = {}
for actuator in ADAM_SP_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  effort = actuator.effort_limit
  stiffness = actuator.stiffness
  for expr in actuator.joint_names_expr:
    if stiffness:
      ADAM_SP_ACTION_SCALE[expr] = 0.25 * effort / stiffness
