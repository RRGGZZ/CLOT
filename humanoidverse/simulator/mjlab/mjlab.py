import sys
import os
from loguru import logger
from pathlib import Path
import torch
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
import numpy as np
import mujoco   
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
# from humanoidverse.simulator.isaaclab_cfg import IsaacLabCfg

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from humanoidverse.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)

from humanoidverse.asset_zoo.robots import (
  ADAM_SP_ACTION_SCALE,
  get_adam_sp_robot_cfg,
)

from humanoidverse.asset_zoo.robots import (
  ADAM_PRO_ACTION_SCALE,
  get_adam_pro_robot_cfg,
)
from mjlab.terrains import TerrainImporterCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation

from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from mjlab.envs import mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.event_manager import EventManager

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.managers.manager_term_config import EventTermCfg
from mjlab.viewer.debug_visualizer import DebugVisualizer
# from mjlab.viewer.offscreen_renderer import OffscreenRenderer

import builtins
import inspect
import copy

class MJlab(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)
        # 从 config 读取 headless 设置，因为 set_headless 是在 base_task 中才被调用的
        if hasattr(config, 'headless'):
            self.headless = config.headless
        self.device = self.sim_device  # 为了兼容 mjlab 的 randomize_field 等函数
        self.simulator_config = config.simulator.config
        self.robot_config = config.robot
        self.env_config = config
        
        self.domain_rand_config = config.domain_rand
        self._body_list = self.robot_config.body_names.copy()
        self.num_envs = self.simulator_config.scene.num_envs
        self.robot_type = getattr(self.robot_config.asset, 'robot_type', 'adam_sp') if hasattr(self.robot_config, 'asset') else "adam_sp"
        
        
        if self.robot_type=="adam_sp" or self.robot_type=="adam_pro": 
            feet_ground_cfg = ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(
                mode="body",
                pattern=r"^(pelvis|hipPitchLeft|hipRollLeft|thighLeft|shinLeft|anklePitchLeft|toeLeft|hipPitchRight|hipRollRight|thighRight|shinRight|anklePitchRight|toeRight|waistRoll_link|waistPitch_link|torso|shoulderPitchLeft|shoulderRollLeft|shoulderYawLeft|elbowLeft|shoulderPitchRight|shoulderRollRight|shoulderYawRight|elbowRight)$",
                entity="robot",
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
                track_air_time=True,
            )
        elif self.robot_type=="g1_23dof_lock_wrist":
            feet_ground_cfg = ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(mode="body", 
                pattern=r"^(pelvis|left_hip_pitch_link|left_hip_roll_link|left_hip_yaw_link|left_knee_link|left_ankle_pitch_link|left_ankle_roll_link|right_hip_pitch_link|right_hip_roll_link|right_hip_yaw_link|right_knee_link|right_ankle_pitch_link|right_ankle_roll_link|waist_yaw_link|waist_roll_link|torso_link|left_shoulder_pitch_link|left_shoulder_roll_link|left_shoulder_yaw_link|left_elbow_link|right_shoulder_pitch_link|right_shoulder_roll_link|right_shoulder_yaw_link|right_elbow_link)$", 
                entity="robot"),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
            )


        self_collision_cfg = ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )

        if self.robot_type == "adam_pro":
            robot_cfg = get_adam_pro_robot_cfg()
        elif self.robot_type == "adam_sp":
            robot_cfg = get_adam_sp_robot_cfg()
        elif self.robot_type == "g1_23dof_lock_wrist":
            robot_cfg = get_g1_robot_cfg()
        else:
            logger.warning(f"Unknown robot_type: {robot_type}, defaulting to adam_sp")
            robot_cfg = get_adam_sp_robot_cfg()

        # print(robot_cfg)
        self.scene_config: SceneCfg = SceneCfg(
            num_envs=self.num_envs, 
            env_spacing=self.simulator_config.scene.env_spacing, 
            terrain=TerrainImporterCfg(terrain_type="plane"), 
            entities={"robot": robot_cfg},
            sensors=(feet_ground_cfg, self_collision_cfg)
        )
        
        mujococfg = MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)
        self.sim_config = SimulationCfg(nconmax=35, njmax=300, mujoco=mujococfg)
        self.view_config = ViewerConfig( origin_type=ViewerConfig.OriginType.ASSET_BODY, asset_name="robot", body_name="pelvis", distance=3.0, elevation=-5.0, azimuth=90.0)
        self.view_config.body_name = "pelvis"
        self.scene = Scene(self.scene_config, device=self.sim_device)

        self.sim = Simulation(num_envs=self.num_envs, cfg=self.sim_config, model=self.scene.compile(), device=self.sim_device)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        # import ipdb; ipdb.set_trace()
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None  # 默认地形是 plane的 hardcode
        # self.scene.entities = {"robot": self.get_robot_cfg()}

        self._robot = self.scene["robot"]
        self.contact_sensor = self.scene["feet_ground_contact"]

        self.mjm = copy.deepcopy(self.sim.mj_model)
        self.mjd = copy.deepcopy(self.sim.mj_data)

        # 初始化暂停标志
        self.is_pause = False

        if not self.headless:
            # 定义键盘回调函数
            def _key_callback(key):
                """处理键盘输入"""
                try:
                    key_char = chr(key) if key < 256 else None
                    
                    # ESC键：关闭viewer
                    if key == 27:  # ESC
                        if self.viewer_handle is not None:
                            self.viewer_handle.close()
                    
                    # 空格键：暂停/继续
                    elif key_char == ' ':
                        self.is_pause = not self.is_pause
                        logger.info(f"仿真 {'已暂停' if self.is_pause else '已继续'}")

                    elif key_char == 'r' or key_char == 'R':
                        if self.viewer_handle is not None:
                            self.viewer_handle.cam.distance = 3.0
                            self.viewer_handle.cam.elevation = -5.0
                            self.viewer_handle.cam.azimuth = 90.0
                            logger.info("视角已重置")
                except (ValueError, OverflowError):
                    pass
            
            self.viewer_handle = mujoco.viewer.launch_passive(
                self.mjm, 
                self.mjd, 
                show_left_ui=True, 
                show_right_ui=True,
                key_callback=_key_callback
            )
            self.viewer_handle.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            
            self.scn = self.viewer_handle.user_scn
            self._vopt = mujoco.MjvOption()
            self._vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            self._pert = mujoco.MjvPerturb()
            self._viz_data = mujoco.MjData(self.mjm)
            self._ghost_model = copy.deepcopy(self.sim.mj_model)
            self._ghost_model.geom_rgba[:] = np.array([0.5, 0.7, 0.5, 0.5])
        else:
            self.viewer_handle = None

        self.load_assets()
        self._base_com_bias = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.friction_coeffs = torch.ones(self.num_envs, 1, 1, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.restitution_coeffs = torch.ones(self.num_envs, 1, 1, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self._link_mass_scale = torch.ones(self.num_envs, len(self.env_config.domain_rand.randomize_link_body_names) if hasattr(self.env_config.domain_rand, 'randomize_link_body_names') else 1, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.pd_gain_scale = torch.ones(self.num_envs, 23, device=self.sim_device, requires_grad=False)
        self.events_cfg: dict[str, EventTermCfg] = {}
        if self.domain_rand_config.get("randomize_friction", False) or self.domain_rand_config.get("randomize_joint_friction", False):
            if self.robot_type == "adam_sp" or self.robot_type == "adam_pro":
                self.events_cfg["foot_friction"] = EventTermCfg(
                    func=mdp.randomize_field,
                    domain_randomization=True,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", geom_names=("toe_left", "toe_right")),
                        "ranges": tuple(self.domain_rand_config["friction_range"]),
                        "operation": "scale",
                        "field": "geom_friction",
                    },
                )
            elif self.robot_type == "g1_23dof_lock_wrist":
                self.events_cfg["foot_friction"] = EventTermCfg(
                    func=mdp.randomize_field,
                    domain_randomization=True,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", geom_names=r"^(left|right)_foot[1-7]_collision$"),
                        "ranges": tuple(self.domain_rand_config["friction_range"]),
                        "operation": "scale",
                        "field": "geom_friction",
                    },
                )

        if self.domain_rand_config.get("randomize_base_com", False):
            if self.robot_type == "adam_sp" or self.robot_type == "adam_pro":
                self.events_cfg["random_base_com"] = EventTermCfg(
                    func=mdp.randomize_field,
                    domain_randomization=True,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            body_names=(
                                "torso",
                        ),
                        ),
                        "ranges": {
                            0: (self.domain_rand_config["base_com_range"]["x"][0], self.domain_rand_config["base_com_range"]["x"][1]),
                            1: (self.domain_rand_config["base_com_range"]["y"][0], self.domain_rand_config["base_com_range"]["y"][1]),
                            2: (self.domain_rand_config["base_com_range"]["z"][0], self.domain_rand_config["base_com_range"]["z"][1]),
                        },
                        "operation": "add",
                        "field": "body_ipos",
                    },
                )
            elif self.robot_type == "g1_23dof_lock_wrist":
                self.events_cfg["random_base_com"] = EventTermCfg(
                    func=mdp.randomize_field,
                    domain_randomization=True,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            body_names=(
                                "torso_link",
                        ),
                        ),
                        "ranges": {
                            0: (self.domain_rand_config["base_com_range"]["x"][0], self.domain_rand_config["base_com_range"]["x"][1]),
                            1: (self.domain_rand_config["base_com_range"]["y"][0], self.domain_rand_config["base_com_range"]["y"][1]),
                            2: (self.domain_rand_config["base_com_range"]["z"][0], self.domain_rand_config["base_com_range"]["z"][1]),
                        },
                        "operation": "add",
                        "field": "body_ipos",
                    },
                )
        if self.domain_rand_config.get("randomize_link_com", False):
            self.events_cfg["random_base_com"] = EventTermCfg(
                func=mdp.randomize_field,
                domain_randomization=True,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=(
                            ".*",
                    ),
                    ),
                    "ranges": {
                        0: (self.domain_rand_config["base_com_range"]["x"][0], self.domain_rand_config["base_com_range"]["x"][1]),
                        1: (self.domain_rand_config["base_com_range"]["y"][0], self.domain_rand_config["base_com_range"]["y"][1]),
                        2: (self.domain_rand_config["base_com_range"]["z"][0], self.domain_rand_config["base_com_range"]["z"][1]),
                    },
                    "operation": "add",
                    "field": "body_ipos",
                },
            )

        if "randomize_link_mass" in self.domain_rand_config and self.domain_rand_config.get("randomize_link_mass", False):
            self.events_cfg["scale_body_mass"] = EventTermCfg(
                func=mdp.randomize_field,
                domain_randomization=True,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "ranges": tuple(self.domain_rand_config["link_mass_range"]),
                    "operation": "scale",
                    "field": "body_mass",
                },
            )
        
        if self.domain_rand_config.get("randomize_pd_gain", False):
            self.events_cfg["randomize_pd_gain"] = EventTermCfg(
                func=mdp.randomize_pd_gains,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "kp_range": tuple(self.domain_rand_config["kp_range"]),
                    "kd_range": tuple(self.domain_rand_config["kd_range"]),
                    "operation": "scale",
                    "distribution": "uniform",
                },
            )
        
        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)
    
        # 收集需要扩展的字段
        fields_to_expand = list(self.event_manager.domain_randomization_fields)
        
        # ⚠️ 重要: randomize_pd_gains需要手动添加actuator相关字段
        # 因为它不是通过randomize_field调用，所以不会被自动检测
        if self.domain_rand_config.get("randomize_pd_gain", False):
            if "actuator_gainprm" not in fields_to_expand:
                fields_to_expand.append("actuator_gainprm")
            if "actuator_biasprm" not in fields_to_expand:
                fields_to_expand.append("actuator_biasprm")
        
        logger.info(f"🔧 Expanding model fields for domain randomization: {fields_to_expand}")
        self.sim.expand_model_fields(tuple(fields_to_expand))

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
#            self.sim.create_graph()  # disabled: causes CUDA error 900
            
            # 验证域随机化（可选）
            logger.info("=" * 60)
            logger.info("验证域随机化是否生效（前5个环境）")
            logger.info("=" * 60)
            
            # 使用全局geom IDs
            # toe_left_local_ids, _ = self._robot.find_geoms("toe_left")
            # toe_right_local_ids, _ = self._robot.find_geoms("toe_right")
            # global_geom_ids = self._robot.indexing.geom_ids.cpu().numpy() if isinstance(self._robot.indexing.geom_ids, torch.Tensor) else self._robot.indexing.geom_ids
            # toe_left_global_id = global_geom_ids[toe_left_local_ids[0]]
            # toe_right_global_id = global_geom_ids[toe_right_local_ids[0]]
            # frictions_left = self.sim.model.geom_friction[:5, toe_left_global_id, 0]
            # frictions_right = self.sim.model.geom_friction[:5, toe_right_global_id, 0]
            # logger.info(f"✅ Friction (toe_left)  : {frictions_left}")
            # logger.info(f"✅ Friction (toe_right) : {frictions_right}")

            # 使用全局body IDs
            # torso_local_ids, _ = self._robot.find_bodies("torso")
            # global_body_ids_torso = self._robot.indexing.body_ids.cpu().numpy() if isinstance(self._robot.indexing.body_ids, torch.Tensor) else self._robot.indexing.body_ids
            # torso_global_id = global_body_ids_torso[torso_local_ids[0]]
            # base_com_x = self.sim.model.body_ipos[:5, torso_global_id, 0]
            # base_com_y = self.sim.model.body_ipos[:5, torso_global_id, 1]
            # base_com_z = self.sim.model.body_ipos[:5, torso_global_id, 2]
            # logger.info(f"✅ Base COM X: {base_com_x}")
            # logger.info(f"✅ Base COM Y: {base_com_y}")
            # logger.info(f"✅ Base COM Z: {base_com_z}")
   
            # 使用全局body IDs  
            body_local_ids, body_names = self._robot.find_bodies(".*")  # 所有body(局部索引)
            global_body_ids_all = self._robot.indexing.body_ids.cpu().numpy() if isinstance(self._robot.indexing.body_ids, torch.Tensor) else self._robot.indexing.body_ids
            if len(body_local_ids) > 0:
                first_body_global_id = global_body_ids_all[body_local_ids[0]]
                link_com_x = self.sim.model.body_ipos[:5, first_body_global_id, 0]
                link_com_y = self.sim.model.body_ipos[:5, first_body_global_id, 1]
                link_com_z = self.sim.model.body_ipos[:5, first_body_global_id, 2]
                logger.info(f"✅ Link COM X ({body_names[0]}): {link_com_x}")
                logger.info(f"✅ Link COM Y ({body_names[0]}): {link_com_y}")
                logger.info(f"✅ Link COM Z ({body_names[0]}): {link_com_z}")
            

            # 使用全局body IDs(而不是本地索引)
            # find_bodies返回的是相对于robot的本地索引
            # 但sim.model.body_mass使用的是全局MuJoCo body ID(包括world, terrain等)
            # 正确的全局IDs存储在self._robot.indexing.body_ids中
            global_body_ids = self._robot.indexing.body_ids.cpu().numpy() if isinstance(self._robot.indexing.body_ids, torch.Tensor) else self._robot.indexing.body_ids
            body_names = self._robot.body_names
            
            logger.info(f"✅ Total robot bodies: {len(body_names)}")
            if len(global_body_ids) >= 3:
                for i in range(min(len(body_names), len(global_body_ids))):
                    global_id = global_body_ids[i]
                    masses = self.sim.model.body_mass[:5, global_id]
                    logger.info(f"✅ Body Mass ({body_names[i]}): {masses}")
            
            joint_ids = self._robot.actuators[0].ctrl_ids
            kp_values = self.sim.model.actuator_gainprm[:5, joint_ids[0], 0]  # 第一个关节
            kd_values = -self.sim.model.actuator_biasprm[:5, joint_ids[0], 2]  # kd是负值
            logger.info(f"✅ PD Kp (joint 0): {kp_values}")
            logger.info(f"✅ PD Kd (joint 0): {kd_values}")
            
            logger.info("=" * 60)

            
        # # 在事件应用后，从物理引擎读取实际的 friction 值并保存
        # if self.domain_rand_config.get("randomize_friction", False):
        #     self._read_friction_coeffs_from_physics()
        # if self.domain_rand_config.get("randomize_restitution", False):
        #     self._read_restitution_coeffs_from_physics()

        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)
        

        self._sim_step_counter = 0

        # debug visualization
        # self.draw = _debug_draw.acquire_debug_draw_interface()
        
        # print the environment information
        logger.info("Completed setting up the environment...")
        
        
    # def _read_friction_coeffs_from_physics(self):
    #     joint_friction = self._robot.root_physx_view.get_dof_friction_coefficients().to(self.sim_device)  # shape: (num_envs, num_dofs)
        
    #     # 取每个环境所有关节的平均值作为该环境的 friction 系数
    #     friction_per_env = joint_friction.mean(dim=1, keepdim=True)  # shape: (num_envs, 1)
        
    #     # 保存为 (num_envs, 1, 1) 格式以匹配 IsaacGym
    #     self.friction_coeffs = friction_per_env.unsqueeze(-1)
        
    #     logger.info(f"从物理引擎读取 friction 系数完成")
    #     logger.info(f"Friction coeffs shape: {self.friction_coeffs.shape}")
    #     logger.info(f"Friction coeffs 范例 (前3个环境): {self.friction_coeffs[:3].flatten()}")
    #     logger.info(f"Friction range: [{self.friction_coeffs.min().item():.4f}, {self.friction_coeffs.max().item():.4f}]")
    
    # def _read_restitution_coeffs_from_physics(self):
    #     # 获取所有环境所有刚体的 restitution 系数
    #     dof_props = self._robot.get_dof_properties().to(self.sim_device)
    #     restitutions = dof_props["restitution"] if "restitution" in dof_props else None  # shape: (num_envs, num_joints)
    #     # shape: (num_envs, num_bodies)
        
    #     # 取每个环境所有刚体的平均值作为该环境的 restitution 系数
    #     restitution_per_env = restitutions.mean(dim=1, keepdim=True)  # shape: (num_envs, 1)
        
    #     # 保存为 (num_envs, 1, 1) 格式以匹配 IsaacGym
    #     self.restitution_coeffs = restitution_per_env.unsqueeze(-1)
        
    #     logger.info(f"从物理引擎读取 restitution 系数完成")
    #     logger.info(f"Restitution coeffs shape: {self.restitution_coeffs.shape}")
    #     logger.info(f"Restitution coeffs 范例 (前3个环境): {self.restitution_coeffs[:3].flatten()}")
    #     logger.info(f"Restitution range: [{self.restitution_coeffs.min().item():.4f}, {self.restitution_coeffs.max().item():.4f}]")


    def set_headless(self, headless):
        # call super
        super().set_headless(headless)
        # if not self.headless:
        #     from isaacsim.util.debug_draw import _debug_draw
        #     self.draw = _debug_draw.acquire_debug_draw_interface()
        # else:
        #     self.draw = None

    def setup(self):
        self.sim_dt = 1. / self.simulator_config.sim.fps
        
    
    def setup_terrain(self, mesh_type):
        pass


    def load_assets(self):
        '''
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        '''

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)

        self.dof_ids, self.dof_names = self._robot.find_joints(dof_names_list, preserve_order=True)
        
        self.body_ids, self.body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)
        print(self.body_ids, self.body_names)

        self._body_list = self.body_names.copy()
        
        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        # warning if the dof_ids order does not match the joint_names order in robot_config
        if self.dof_ids != list(range(self.num_dof)):
            logger.warning("The order of the joint_names in the robot_config does not match the order of the joint_ids in IsaacSim.")
        
        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), "Number of bodies must be equal to number of body names"
        # import ipdb; ipdb.set_trace()
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"
       
        
        # return self.num_dof, self.num_bodies, self.dof_names, self.body_names
        

    def create_envs(self, num_envs, env_origins, base_init_state):
        
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state
        
        return self.scene, self._robot
    
    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.env_config.rewards.reward_limit.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        '''
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order
        '''
        indices, names = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        elif len(indices) == 1:
            return indices[0]
        else: # multiple bodies found
            logger.warning(f"Multiple bodies found for {body_name}.")
            return indices
    
    def prepare_sim(self):
        self.refresh_sim_tensors() # initialize tensors

    @property
    def dof_state(self):
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self):
        self.all_root_states = torch.cat([self._robot.data.root_link_pose_w, self._robot.data.root_link_vel_w], dim=-1)  # (num_envs, 13)
        self.robot_root_states = self.all_root_states # (num_envs, 13)
        self.base_quat = self.robot_root_states[:, [4, 5, 6, 3]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        # import ipdb; ipdb.set_trace()
        
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids] # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        self.contact_forces =  - self.contact_sensor.data.force # (num_envs, num_bodies, 3)

        self._rigid_body_pos = self._robot.data.body_link_pos_w[:, self.body_ids, :]
        self._rigid_body_rot = self._robot.data.body_link_quat_w[:, self.body_ids][:, :, [1, 2, 3, 0]] # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        self._rigid_body_vel = self._robot.data.body_link_lin_vel_w[:, self.body_ids, :]
        self._rigid_body_ang_vel = self._robot.data.body_link_ang_vel_w[:, self.body_ids, :]

        # import ipdb; ipdb.set_trace()


    def apply_actions(self, actions):
        self._robot.set_joint_position_target(actions)
    
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        # import ipdb; ipdb.set_trace()
        self._robot.write_root_state_to_sim(root_states[set_env_ids, :], set_env_ids)

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        dof_pos, dof_vel = dof_states[set_env_ids, :, 0], dof_states[set_env_ids, :, 1]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, set_env_ids)
        self._robot.clear_state(env_ids=set_env_ids)
        # import ipdb; ipdb.set_trace()
    
    def simulate_at_each_physics_step(self):
        # 如果暂停，等待直到继续
        if hasattr(self, 'is_pause') and self.is_pause:
            import time
            time.sleep(0.01)  # 短暂休眠，避免CPU占用过高
            return
        
        self._sim_step_counter += 1
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(dt=1./self.simulator_config.sim.fps)
    
    def setup_viewer(self):
        self.viewer = True


    def render(self, sync_frame_time=True):
        if self.headless or self.viewer_handle is None:
            return
        sim_data = self.sim.data
        self.mjd.qpos[:] = sim_data.qpos[0].clone().cpu().numpy()
        self.mjd.qvel[:] = sim_data.qvel[0].cpu().numpy()
        # if self.mjm.nmocap > 0:
        #     self.mjd.mocap_pos[:] = sim_data.mocap_pos[0].cpu().numpy()
        #     self.mjd.mocap_quat[:] = sim_data.mocap_quat[0].cpu().numpy()
        
        mujoco.mj_forward(self.mjm, self.mjd)
        self.viewer_handle.sync()
    
    def close(self) -> None:
        if self.viewer_handle is not None and self.viewer_handle.is_running():
            self.viewer_handle.close()

    #  # debug visualization
    # def clear_lines(self):
    #     self.draw.clear_lines()
    #     self.draw.clear_points()

    # def draw_sphere(self, pos, radius, color, env_id, pos_id):
    #     # draw a big sphere
    #     point_list = [(pos[0].item(), pos[1].item(), pos[2].item())]
    #     color_list = [(color[0], color[1], color[2], 1.0)]
    #     sizes = [20]
    #     self.draw.draw_points(point_list, color_list, sizes)

    # def draw_line(self, start_point, end_point, color, env_id):
    #     # import ipdb; ipdb.set_trace()
    #     start_point_list = [(   start_point.x.item(), start_point.y.item(), start_point.z.item())]
    #     end_point_list = [(end_point.x.item(), end_point.y.item(), end_point.z.item())]
    #     color_list = [(color.x, color.y, color.z, 1.0)]
    #     sizes = [1]
    #     self.draw.draw_lines(start_point_list, end_point_list, color_list, sizes)
     # debug visualization
    
    def debug_draw(self, qpos):

        qpos = qpos.cpu().numpy()
        self._viz_data.qpos[:] = qpos
        mujoco.mj_forward(self._ghost_model, self._viz_data)
        mujoco.mjv_addGeoms(self._ghost_model, self._viz_data, self._vopt, self._pert, mujoco.mjtCatBit.mjCAT_DYNAMIC.value, self.scn,)

    def clear(self):
            self.scn.ngeom = 0

    def draw_sphere(self, pos, radius, color, env_id, pos_id):
        pass

    def draw_line(self, start_point, end_point, color, env_id):
        pass


