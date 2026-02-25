"""
Adam SP Robot Configuration for Isaac Lab copy pasted from Adam SP BeyondMimic.

Actuator Specifications:
========================
# PND-130-92-7-P (Hip/Knee Pitch - Planetary Reducer)
# Max joint velocity: 180 RPM (~18.85 rad/s)
# Max torque: 340 N·m
# Rotor inertia: J_m=2.74×10^(−3) kg·m²
# Gear ratio: N=7
# J_ref = N²×J_m = 7²×2.74×10^(−3) = 0.13426 kg·m²

# PND-50-14A-50-S (Shoulder Pitch/Roll - Harmonic Reducer)
# Max joint velocity: 47 RPM (~4.92 rad/s)
# Max torque: 60 N·m
# Rotor inertia: J_m=6.07×10^(−5) kg·m²
# Gear ratio: N=51
# J_ref = N²×J_m = 51²×6.07×10^(−5) = 0.1578807 kg·m²

# PND-30-14A-50-S (Shoulder Yaw, Elbow - Harmonic Reducer)
# Max joint velocity: 47 RPM (~4.92 rad/s)
# Max torque: 17.5 N·m
# Rotor inertia: J_m=1.63×10^(−5) kg·m²
# Gear ratio: N=51
# J_ref = N²×J_m = 51²×1.63×10^(−5) = 0.0423963 kg·m²

# PND-60-17-50-S (Hip Yaw, Waist - Harmonic Reducer)
# Max joint velocity: 40 RPM (~4.19 rad/s)
# Max torque: 89 N·m
# Rotor inertia: J_m=9×10^(-5) kg·m²
# Gear ratio: N=51
# J_ref = N²×J_m = 51²×9×10^(-5) = 0.23409 kg·m²

# PND-80-20-30-S (Hip Roll - Harmonic Reducer)
# Max joint velocity: 80 RPM (~8.38 rad/s)
# Max torque: 120 N·m
# Rotor inertia: J_m=2.93×10^(−4) kg·m²
# Gear ratio: N=31
# J_ref = N²×J_m = 31²×2.93×10^(−4) = 0.281573 kg·m²

# PND-50-52-30-P (Ankle - Planetary Reducer)
# Max joint velocity: 80 RPM (~8.38 rad/s)
# Max torque: 46 N·m
# Rotor inertia: J_m=6.1×10^(−5) kg·m²
# Gear ratio: N=30
# J_ref = N²×J_m = 30²×6.1×10^(−5) = 0.0549 kg·m²

Joint to Actuator Mapping:
==========================
shoulderPitch_Left/Right  -> PND-50-14A-50-S
shoulderRoll_Left/Right   -> PND-50-14A-50-S
shoulderYaw_Left/Right    -> PND-30-14A-50-S
elbow_Left/Right          -> PND-30-14A-50-S
waistRoll/Pitch/Yaw       -> PND-60-17-50-S
hipPitch_Left/Right       -> PND-130-92-7-P
hipRoll_Left/Right        -> PND-80-20-30-S
hipYaw_Left/Right         -> PND-60-17-50-S
kneePitch_Left/Right      -> PND-130-92-7-P
anklePitch_Left/Right     -> PND-50-52-30-P
ankleRoll_Left/Right      -> PND-50-52-30-P
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Armature values (reflected inertia J_ref)
ARMATURE_130_92_7_P = 0.13426      # Hip/Knee Pitch (PND-130-92-7-P)
ARMATURE_50_14A_50_S = 0.1578807   # Shoulder Pitch/Roll (PND-50-14A-50-S)
ARMATURE_30_14A_50_S = 0.0423963   # Shoulder Yaw, Elbow (PND-30-14A-50-S)
ARMATURE_60_17_50_S = 0.23409     # Hip Yaw, Waist (PND-60-17-50-S)
ARMATURE_80_20_30_S = 0.281573    # Hip Roll (PND-80-20-30-S)
ARMATURE_50_52_30_P = 0.0549      # Ankle (PND-50-52-30-P)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# Stiffness = Armature * omega^2
STIFFNESS_130_92_7_P = ARMATURE_130_92_7_P * NATURAL_FREQ**2
STIFFNESS_50_14A_50_S = ARMATURE_50_14A_50_S * NATURAL_FREQ**2
STIFFNESS_30_14A_50_S = ARMATURE_30_14A_50_S * NATURAL_FREQ**2
STIFFNESS_60_17_50_S = ARMATURE_60_17_50_S * NATURAL_FREQ**2
STIFFNESS_80_20_30_S = ARMATURE_80_20_30_S * NATURAL_FREQ**2
STIFFNESS_50_52_30_P = ARMATURE_50_52_30_P * NATURAL_FREQ**2

# Damping = 2 * zeta * Armature * omega
DAMPING_130_92_7_P = 2.0 * DAMPING_RATIO * ARMATURE_130_92_7_P * NATURAL_FREQ
DAMPING_50_14A_50_S = 2.0 * DAMPING_RATIO * ARMATURE_50_14A_50_S * NATURAL_FREQ
DAMPING_30_14A_50_S = 2.0 * DAMPING_RATIO * ARMATURE_30_14A_50_S * NATURAL_FREQ
DAMPING_60_17_50_S = 2.0 * DAMPING_RATIO * ARMATURE_60_17_50_S * NATURAL_FREQ
DAMPING_80_20_30_S = 2.0 * DAMPING_RATIO * ARMATURE_80_20_30_S * NATURAL_FREQ
DAMPING_50_52_30_P = 2.0 * DAMPING_RATIO * ARMATURE_50_52_30_P * NATURAL_FREQ

ADAM_SP_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/adam_sp/urdf/adam_sp.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.90),  # x, y, z [m]
        rot=(0.0, 0.0, 0.0, 1.0),  # x, y, z, w [quat]
        lin_vel=(0.0, 0.0, 0.0),  # x, y, z [m/s]
        ang_vel=(0.0, 0.0, 0.0),  # x, y, z [rad/s]
        joint_pos={
            # Left leg
            "hipPitch_Left": -0.32,
            "hipRoll_Left": 0.0,
            "hipYaw_Left": -0.18,
            "kneePitch_Left": 0.66,
            "anklePitch_Left": -0.39,
            "ankleRoll_Left": -0.0,
            # Right leg
            "hipPitch_Right": -0.32,
            "hipRoll_Right": -0.0,
            "hipYaw_Right": 0.18,
            "kneePitch_Right": 0.66,
            "anklePitch_Right": -0.39,
            "ankleRoll_Right": -0.0,
            # Waist
            "waistRoll": 0.0,
            "waistPitch": 0.0,
            "waistYaw": 0.0,
            # Left arm
            "shoulderPitch_Left": 0.0,
            "shoulderRoll_Left": 0.1,
            "shoulderYaw_Left": 0.0,
            "elbow_Left": -0.3,
            # Right arm
            "shoulderPitch_Right": 0.0,
            "shoulderRoll_Right": -0.1,
            "shoulderYaw_Right": 0.0,
            "elbow_Right": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_pitch": ImplicitActuatorCfg(
            joint_names_expr=[
                "hipPitch_.*",
                "kneePitch_.*",
            ],
            effort_limit_sim={
                "hipPitch_.*": 340.0,
                "kneePitch_.*": 340.0,
            },
            velocity_limit_sim={
                "hipPitch_.*": 18.85,  # 180 RPM
                "kneePitch_.*": 18.85,
            },
            stiffness={
                "hipPitch_.*": STIFFNESS_130_92_7_P,
                "kneePitch_.*": STIFFNESS_130_92_7_P,
            },
            damping={
                "hipPitch_.*": DAMPING_130_92_7_P,
                "kneePitch_.*": DAMPING_130_92_7_P,
            },
            armature={
                "hipPitch_.*": ARMATURE_130_92_7_P,
                "kneePitch_.*": ARMATURE_130_92_7_P,
            },
        ),
        "legs_roll": ImplicitActuatorCfg(
            joint_names_expr=["hipRoll_.*"],
            effort_limit_sim=120.0,
            velocity_limit_sim=8.38,  # 80 RPM
            stiffness=STIFFNESS_80_20_30_S,
            damping=DAMPING_80_20_30_S,
            armature=ARMATURE_80_20_30_S,
        ),
        "legs_yaw": ImplicitActuatorCfg(
            joint_names_expr=["hipYaw_.*"],
            effort_limit_sim=89.0,
            velocity_limit_sim=4.19,  # 40 RPM
            stiffness=STIFFNESS_60_17_50_S,
            damping=DAMPING_60_17_50_S,
            armature=ARMATURE_60_17_50_S,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["anklePitch_.*", "ankleRoll_.*"],
            effort_limit_sim=46.0,
            velocity_limit_sim=8.38,  # 80 RPM
            stiffness=STIFFNESS_50_52_30_P,
            damping=DAMPING_50_52_30_P,
            armature=ARMATURE_50_52_30_P,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waistRoll", "waistPitch", "waistYaw"],
            effort_limit_sim=89.0,
            velocity_limit_sim=4.19,  # 40 RPM
            stiffness=STIFFNESS_60_17_50_S,
            damping=DAMPING_60_17_50_S,
            armature=ARMATURE_60_17_50_S,
        ),
        "arms_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulderPitch_.*",
                "shoulderRoll_.*",
            ],
            effort_limit_sim={
                "shoulderPitch_.*": 60.0,
                "shoulderRoll_.*": 60.0,
            },
            velocity_limit_sim={
                "shoulderPitch_.*": 4.92,  # 47 RPM
                "shoulderRoll_.*": 4.92,
            },
            stiffness={
                "shoulderPitch_.*": STIFFNESS_50_14A_50_S,
                "shoulderRoll_.*": STIFFNESS_50_14A_50_S,
            },
            damping={
                "shoulderPitch_.*": DAMPING_50_14A_50_S,
                "shoulderRoll_.*": DAMPING_50_14A_50_S,
            },
            armature={
                "shoulderPitch_.*": ARMATURE_50_14A_50_S,
                "shoulderRoll_.*": ARMATURE_50_14A_50_S,
            },
        ),
        "arms_elbow": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulderYaw_.*",
                "elbow_.*",
            ],
            effort_limit_sim={
                "shoulderYaw_.*": 17.5,
                "elbow_.*": 17.5,
            },
            velocity_limit_sim={
                "shoulderYaw_.*": 4.92,  # 47 RPM
                "elbow_.*": 4.92,
            },
            stiffness={
                "shoulderYaw_.*": STIFFNESS_30_14A_50_S,
                "elbow_.*": STIFFNESS_30_14A_50_S,
            },
            damping={
                "shoulderYaw_.*": DAMPING_30_14A_50_S,
                "elbow_.*": DAMPING_30_14A_50_S,
            },
            armature={
                "shoulderYaw_.*": ARMATURE_30_14A_50_S,
                "elbow_.*": ARMATURE_30_14A_50_S,
            },
        ),
    },
)

ADAM_SP_ACTION_SCALE = {}
for a in ADAM_SP_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            ADAM_SP_ACTION_SCALE[n] = 0.25 * e[n] / s[n]



