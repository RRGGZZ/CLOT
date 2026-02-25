
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


ARTICULATION_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path="humanoidverse/data/robots/adam_sp/adam_sp.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "hipPitch_Left": -0.586,
            "hipRoll_Left": -0.085,
            "hipYaw_Left": -0.322,
            "kneePitch_Left": 1.288,
            "anklePitch_Left": -0.789,
            "ankleRoll_Left": 0.002,
            "hipPitch_Right": -0.586,
            "hipRoll_Right": 0.085,
            "hipYaw_Right": 0.322,
            "kneePitch_Right": 1.288,
            "anklePitch_Right": -0.789,
            "ankleRoll_Right": -0.002,
            "waistRoll": 0.0,
            "waistPitch": 0.0,
            "waistYaw": 0.0,
            "shoulderPitch_Left": 0.0,
            "shoulderRoll_Left": 0.0,
            "shoulderYaw_Left": 0.0,
            "elbow_Left": -0.3,
            "shoulderPitch_Right": 0.0,
            "shoulderRoll_Right": 0.0,
            "shoulderYaw_Right": 0.0,
            "elbow_Right": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,        
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["hipPitch_.*", "hipRoll_.*", "hipYaw_.*", "kneePitch_.*", "anklePitch_.*", "ankleRoll_.*"],
            effort_limit_sim={
                "hipPitch_.*": 230.0,
                "hipRoll_.*": 160.0,
                "hipYaw_.*": 105.0,
                "kneePitch_.*": 230.0,
                "anklePitch_.*": 40.0,
                "ankleRoll_.*": 12.0,
            },
            velocity_limit_sim={
                "hipPitch_.*": 15.0,
                "hipRoll_.*": 8.0,
                "hipYaw_.*": 8.0,
                "kneePitch_.*": 15.0,
                "anklePitch_.*": 20.0,
                "ankleRoll_.*": 20.0,
            },
            stiffness={
                "hipPitch_.*": 305.0,
                "hipRoll_.*": 255.0,
                "hipYaw_.*": 255.0,
                "kneePitch_.*": 305.0,
                "anklePitch_.*": 50.0,
                "ankleRoll_.*": 30.0,
            },
            damping={
                "hipPitch_.*": 5.0,
                "hipRoll_.*": 3.5,
                "hipYaw_.*": 3.5,
                "kneePitch_.*": 5.0,
                "anklePitch_.*": 0.8,
                "ankleRoll_.*": 0.35,
            },
            armature={
                "hipPitch_.*": 0.03,
                "hipRoll_.*": 0.03,
                "hipYaw_.*": 0.03,
                "kneePitch_.*": 0.03,
                "anklePitch_.*": 0.03,
                "ankleRoll_.*": 0.03,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waistRoll", "waistPitch", "waistYaw"],
            effort_limit_sim={
                "waistRoll": 110.0,
                "waistPitch": 110.0,
                "waistYaw": 110.0,
            },
            velocity_limit_sim={
                "waistRoll": 8.0,
                "waistPitch": 8.0,
                "waistYaw": 8.0,
            },
            stiffness={
                "waistRoll": 255.0,
                "waistPitch": 305.0,
                "waistYaw": 255.0,
            },
            damping={
                "waistRoll": 3.5,
                "waistPitch": 5.0,
                "waistYaw": 3.5,
            },
            armature={
                "waistRoll": 0.03,
                "waistPitch": 0.03,
                "waistYaw": 0.03,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["shoulderPitch_.*", "shoulderRoll_.*", "shoulderYaw_.*", "elbow_.*"],
            effort_limit_sim={
                "shoulderPitch_.*": 65.0,
                "shoulderRoll_.*": 65.0,
                "shoulderYaw_.*": 65.0,
                "elbow_.*": 30.0,
            },
            velocity_limit_sim={
                "shoulderPitch_.*": 8.0,
                "shoulderRoll_.*": 8.0,
                "shoulderYaw_.*": 8.0,
                "elbow_.*": 8.0,
            },
            stiffness={
                "shoulderPitch_.*": 40.0,
                "shoulderRoll_.*": 40.0,
                "shoulderYaw_.*": 40.0,
                "elbow_.*": 40.0,
            },
            damping={
                "shoulderPitch_.*": 1.0,
                "shoulderRoll_.*": 1.0,
                "shoulderYaw_.*": 1.0,
                "elbow_.*": 1.0,
            },
            armature={
                "shoulderPitch_.*": 0.03,
                "shoulderRoll_.*": 0.03,
                "shoulderYaw_.*": 0.03,
                "elbow_.*": 0.03,
            },
        ),
    },
)

ADAM_SP_ACTION_SCALE = {}
for a in ARTICULATION_CFG.actuators.values():
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