from typing import Dict, List, Tuple, Callable, Optional, Any, Union, TypeVar
import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
from loguru import logger
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.helpers import parse_observation, np2torch, torch2np
from humanoidverse.utils.motion_lib.motion_lib_robot_ztj import MotionLibRobotZTJ as MotionLibRobot
from humanoidverse.utils.lpf import ActionFilterButterTorch

import time
import torch
import os
from pathlib import Path
# from description.robots.dtype import RobotExitException
from isaac_utils.rotations import calc_heading_quat_inv, my_quat_rotate, calc_yaw_heading_quat_inv

# 全局变量：是否从motion_lib读取初始姿态
USE_MOTION_LIB_INIT = True  

URCIRobotType = TypeVar('URCIRobotType', bound='URCIRobot')
ObsCfg = Union[DictConfig, Callable[[URCIRobotType],np.ndarray]]
URCIPolicyObs = Tuple[ObsCfg, Callable]

CfgType = Union[OmegaConf, ListConfig, DictConfig]


def wrap_to_pi_float(angles:float):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

class URCIRobot:
    REAL: bool
    BYPASS_ACT: bool
    SWITCH_EMA: bool
    
    dt: float # big dt, not small dt
    clip_observations: float
    cfg: OmegaConf
    
    q: np.ndarray
    dq: np.ndarray
    # pos: np.ndarray
    # vel: np.ndarray
    quat: np.ndarray  # XYZW
    omega: np.ndarray
    gvec: np.ndarray
    rpy: np.ndarray
    
    act: np.ndarray
    
    _obs_cfg_obs: ObsCfg
    # _obs_cfg_obs: Optional[ObsCfg]=None
    _ref_pid = -2 # reference policy id
                    # Normal pid: (0,1,2,3)
                    # Special pid: (-2, -3, -4)
    _pid_size = -1
    
    def __init__(self, cfg: CfgType):
        self.BYPASS_ACT = cfg.deploy.BYPASS_ACT
        self.SWITCH_EMA = cfg.deploy.SWITCH_EMA
        self.counter = 0
        self.track_ref_accumulated = {
            'ref_dof_pos': [],
            'ref_body_pos_extend': [],
            'dof_pos': [],
            'body_pos_extend': [],
            'dif_local_rigid_body_pos': [],
            'counter': []
        }
        # self.save_dir = 'logs/track_data_without_advance'
        # os.makedirs(self.save_dir, exist_ok=True)

        
        self.cfg: OmegaConf = cfg
        self._obs_cfg_obs = cfg.obs
        self.device: str = "cpu"
        self.dt: float = cfg.deploy.ctrl_dt
        self.timer: int = 0
        self.motion_start_idx: int = 0
        
        self.num_actions = cfg.robot.actions_dim
        self.heading_cmd = cfg.deploy.heading_cmd   
        self.clip_action_limit: float = cfg.robot.control.action_clip_value
        self.clip_observations: float = cfg.env.config.normalization.clip_observations
        self.effort_limit = np.array(cfg.robot.dof_effort_limit_list)
        # num_rigid_bodies_extend = num_bodies + nums_extend_bodies (e.g. G1: 24+3=27, Adam: 24+3=27)
        num_bodies = OmegaConf.select(cfg, "robot.num_bodies", default=24)
        nums_extend_bodies = OmegaConf.select(cfg, "robot.motion.nums_extend_bodies", default=3)
        self.num_rigid_bodies_extend = num_bodies + nums_extend_bodies
        # self.action_scale: float = cfg.robot.control.action_scale
        if "action_filt" in cfg.robot.control and cfg.robot.control.action_filt:
            self.action_filter = ActionFilterButterTorch(lowcut=np.zeros(self.num_actions),
                                                        highcut=np.ones(self.num_actions) * cfg.robot.control.action_cutfreq, 
                                                        sampling_rate=1./self.dt, num_joints=self.num_actions, 
                                                        device=self.device)

        if cfg.log_task_name == "motion_tracking":
            self.is_motion_tracking = True
            # 在初始化时就创建motion_libs，而不是传入空列表
            self._make_motionlib_from_config()
        else:
            self.is_motion_tracking = False
        
        self._make_init_pose()
        self._make_buffer()
        self.action_scale = 0.25 * self.effort_limit / self.kp
        # if 'save_motion' in cfg.env.config:
        #     self.save_motion = bool(cfg.env.config.save_motion)
        #     if self.save_motion: self._make_init_save_motion()
        # else: 
        #     self.save_motion = False
        
    
    def routing(self, cfg_policies: List[URCIPolicyObs]):
        """
            Usage: Input a list of Policy, and the robot can switch between them.
            
            - Policies are indexed by integers (Pid), 0 to len(cfg_policies)-1.
            - special pid: 
                - -2: Reset the robot.
                - 0: Default policy, should be stationary. 
                    - The Robot will switch to this policy once the motion tracking is Done or being Reset.
            - Switching Mechanism:
                - The instance (MuJoCo or Real) should implement the Pid control logic. It can be changed at any time.
                - When the instance want to Reset the robot, it should set the pid to -2.
        """
        self._pid_size = len(cfg_policies)
        
        # 只在需要时更新motion_libs（如果策略数量与现有motion_libs数量不匹配）
        if len(self.motion_libs) != len(cfg_policies):
            logger.info(f"Updating motion_libs: {len(self.motion_libs)} -> {len(cfg_policies)}")
            self._make_motionlib(cfg_policies)
        else:
            logger.info(f"Using existing motion_libs: {len(self.motion_libs)}")
            
        self._check_init()
        self.cmd[3]=self.rpy[2]
        cur_pid = -1

        # try: 
        while True:
            t1 = time.time()
            
            if cur_pid != self._ref_pid or self._ref_pid == -2:
                if self._ref_pid == -2:
                    self.Reset()
                    self._ref_pid = 0
                    t1 = time.time()
                    ...
                
                
                self._ref_pid %= self._pid_size
                assert self._ref_pid >= 0 and self._ref_pid < self._pid_size, f"Invalid policy id: {self._ref_pid}"
                # self.TrySaveMotionFile(pid=cur_pid)       
                logger.info(f"Switch to the policy {self._ref_pid}")

                
                cur_pid = self._ref_pid
                self.SetObsCfg(cfg_policies[cur_pid][0])
                policy_fn = cfg_policies[cur_pid][1]
                if self.SWITCH_EMA:
                    self.old_act = self.act.copy()
                # print('Debug: ',self.Obs()['actor_obs'])
                # breakpoint()

                
                # breakpoint()
            
            self.UpdateObs()
            
            action = policy_fn(self.Obs())[0]
            # print(f"action: {action}")
            
            if self.BYPASS_ACT: action = np.zeros_like(action)
            
            # if self.SWITCH_EMA and self.timer <10:
            #     self.old_act = self.old_act * 0.9 + action * 0.1
            #     action = self.old_act
                
            
            self.ApplyAction(action)
            
            
            # self.TrySaveMotionStep()
            
            if self.motion_len > 0 and self.ref_motion_phase > 1.0:
                # self.Reset()
                if self._ref_pid == 0:
                    self._ref_pid = -2
                else:
                    self._ref_pid = 0
                # self.TrySaveMotionFile(pid=cur_pid)
                logger.info("Motion End. Switch to the Default Policy")
            t2 = time.time()
            
            # print(f"t2-t1 = {(t2-t1)*1e3} ms")
            if self.REAL:
            # if True:
                # print(f"t2-t1 = {(t2-t1)*1e3} ms")
                remain_dt = self.dt - (t2-t1)
                if remain_dt > 0:
                    time.sleep(remain_dt)
                else:
                    logger.warning(f"Warning! delay = {t2-t1} longer than policy_dt = {self.dt} , skip sleeping")
        # except RobotExitException as e:
        #     self.TrySaveMotionFile(pid=cur_pid)
        #     raise e
        # ...
    
    def _reset(self):
        raise NotImplementedError("Not implemented")
    
    def Reset(self):
        # self.TrySaveMotionFile()
        self._reset()
        
        self.act[:] = 0
        self.history_handler.reset([0])
        self.timer: int = 0
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)
        self.cmd[3]=self.rpy[2]
        
        
        self.UpdateObs()
        
    def _apply_action(self, target_q):
        raise NotImplementedError("Not implemented")
    
    # @_prof_applyaction
    def ApplyAction(self, action): 
        self.timer += 1
        
        self.act = np.clip(action, -self.clip_action_limit, self.clip_action_limit)
        action = np.clip(action, -self.clip_action_limit, self.clip_action_limit)
        # logger.info(f"action: {action}")
        if "action_filt" in self.cfg.robot.control and self.cfg.robot.control.action_filt:
            action = torch.tensor(action)
            action = self.action_filter.filter(action)
            action = np.array(action)
        target_q = action * self.action_scale + self.dof_init_pose
        
        self._apply_action(target_q)
        

    
    def Obs(self)->Dict[str, np.ndarray]:
        if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
            return {'actor_obs_current': torch2np(self.obs_buf_dict['actor_obs_current']),
                    'actor_obs_past': torch2np(self.obs_buf_dict['actor_obs_past']),
                    'actor_obs_future': torch2np(self.obs_buf_dict['actor_obs_future'])}
        else:
            return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}
    
    
    def _get_state(self):
        raise NotImplementedError("Not implemented")
    
    def GetState(self):
        self._get_state()
        
        
        if self.heading_cmd:
            self.cmd[2] = np.clip(0.5*wrap_to_pi_float(self.cmd[3] - self.rpy[2]), -1., 1.)
            
        if self.is_motion_tracking:
            self.motion_time = (self.timer) * self.dt 
            self.ref_motion_phase = self.motion_time / self.motion_len
        print("cmd: ", self.cmd,end='\r\b')
    

    def SetObsCfg(self, obs_cfg: ObsCfg):
        if isinstance(obs_cfg, DictConfig):
            self._obs_cfg_obs = obs_cfg
            self.motion_len = self._obs_cfg_obs.motion_len
        elif isinstance(obs_cfg, Callable):
            self._obs_cfg_obs = obs_cfg
            self.motion_len = -1
        else:
            raise NotImplementedError("Not implemented")
        
        # self.act[:] = 0
        # self.history_handler.reset([0])
        # 确保motion_lib索引有效
        if len(self.motion_libs) > 0:
            # 如果只有一个motion_lib，始终使用索引0
            if len(self.motion_libs) == 1:
                self.motion_lib = self.motion_libs[0]
            else:
                # 如果有多个motion_lib，使用策略索引
                self.motion_lib = self.motion_libs[self._ref_pid]
        else:
            self.motion_lib = None
        
        self.motion_length = self.motion_lib.get_motion_length(self.motion_ids)
        self.motion_dt = self.motion_lib._motion_dt[self.motion_ids]
        self.timer: int = 0
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)
        self.cmd[3]=self.rpy[2]
        self.ref_init_yaw[0] = self.rpy[2]
        if 'ref_motion_phase' in self.history_handler.history.keys():
            self.ref_motion_phase = 0.
            self.history_handler.history['ref_motion_phase']*=0
        self.KickMotionLib()
        # self.UpdateObsWoHistory() # Recompute obs with new _obs_cfg_obs
        ...
    
    def UpdateObsWoHistory(self):
        # if isinstance(self._obs_cfg_obs, Callable):
        #     # self.obs_buf_dict = dict()
        #     self.obs_buf_dict['actor_obs'] = self._obs_cfg_obs(self)
        #     return 
        
        obs_cfg_obs = self._obs_cfg_obs
        
        self.obs_buf_dict_raw = {}
        
        noise_extra_scale = 0.
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict_raw[obs_key] = dict()

            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], obs_cfg_obs.obs_scales, obs_cfg_obs.noise_scales, noise_extra_scale)
        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            obs_keys = sorted(obs_config)
            # (Pdb) sorted(obs_config)
            # ['actions', 'base_ang_vel', 'base_lin_vel', 'dif_local_rigid_body_pos', 'dof_pos', 'dof_vel', 'dr_base_com', 'dr_ctrl_delay', 'dr_friction', 'dr_kd', 'dr_kp', 'dr_link_mass', 'history_critic', 'local_ref_rigid_body_pos', 'projected_gravity', 'ref_motion_phase']
            

            # print("obs shape:", {key: self.obs_buf_dict_raw[obs_key][key].shape for key in obs_keys})
            # for key in obs_keys:
            #     logger.info('obs value {}:{}', key, self.obs_buf_dict_raw[obs_key][key])  
            if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
                self.obs_buf_dict[obs_key+'_current'] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys if 'history' not in key and 'future' not in key], dim=-1).unsqueeze(1)
                self.obs_buf_dict[obs_key+'_past'] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys if 'history' in key], dim=-1)
                self.obs_buf_dict[obs_key+'_future'] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys if 'future' in key], dim=-1)   
            else:
                self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)
            
            
        clip_obs = self.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)
        # breakpoint()

    def UpdateObsForHistory(self):
        hist_cfg_obs = self.cfg.obs
        
        self.hist_obs_dict = {}
        
        noise_extra_scale = 0.
        # Compute history observations
        history_obs_list = self.history_handler.history.keys()
        # print(f"history_obs_list: {history_obs_list}")
        parse_observation(self, history_obs_list, self.hist_obs_dict, hist_cfg_obs.obs_scales, hist_cfg_obs.noise_scales, noise_extra_scale)
        
        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])
    
    _kick_motion_res_counter = -1
    _kick_motion_res_buffer: Optional[Dict[str, torch.Tensor]] = None
    def _kick_motion_res(self)->Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter == self.timer:
            return self._kick_motion_res_buffer # type: ignore
        
        self._kick_motion_res_counter = self.timer
        
        motion_times = torch.tensor((self.timer+1) * self.dt, dtype=torch.float32)
        self._kick_motion_res_buffer = self.motion_lib.get_motion_state(self.motion_ids, motion_times)
        
        return self._kick_motion_res_buffer
    
    _kick_motion_res_counter_multistep = -1
    _kick_motion_res_multistep_buffer: Optional[Dict[str, torch.Tensor]] = None
    def kick_motion_res_multiplestep(self) -> Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter_multistep == self.timer:
            return self._kick_motion_res_multistep_buffer # type: ignore
        self._kick_motion_res_counter_multistep = self.timer
        motion_times = torch.tensor([((self.timer+1) * self.dt  + i * self.dt) for i in range(self.cfg.obs.future_ref_steps)])
        motion_ids = self.motion_ids.repeat(self.cfg.obs.future_ref_steps)
        motion = self.motion_lib.get_motion_state(motion_ids, motion_times)
        self._kick_motion_res_multistep_buffer = {k: v.unsqueeze(0) for k, v in motion.items()}
        return self._kick_motion_res_multistep_buffer
    
    
    # @_prof_getmotion
    def KickMotionLib(self):
        # motion_time x motion lib -> ref state  for obs
        if self.motion_lib is None:
            return
        motion_res =  self._kick_motion_res()
            # (Pdb) motion_res.keys()
            # dict_keys(['contact_mask', 'root_pos', 'root_rot', 'dof_pos', 'root_vel', 'root_ang_vel', 'dof_vel', 'motion_aa', 'motion_bodies', 'rg_pos', 'rb_rot', 'body_vel', 'body_ang_vel', 'rg_pos_t', 'rg_rot_t', 'body_vel_t', 'body_ang_vel_t'])

        current_yaw = self.rpy[2]
        self.relyaw = current_yaw - self.ref_init_yaw
        
        relyaw_heading_inv_quat = calc_yaw_heading_quat_inv(torch.from_numpy(self.relyaw).to(dtype=torch.float32).unsqueeze(0))
        relyaw_heading_inv_quat_expand = relyaw_heading_inv_quat.unsqueeze(1).expand(-1, self.num_rigid_bodies_extend, -1).reshape(-1, 4)

        heading_inv_rot = calc_heading_quat_inv(torch.from_numpy(self.quat).to(dtype=torch.float32).unsqueeze(0), w_last=True) #xyzw
        self.heading_inv_rot = heading_inv_rot
        # # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, self.num_rigid_bodies_extend, -1).reshape(-1, 4)


        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        ref_body_pos_extend = motion_res["rg_pos_t"]
        
        self.ref_motion = ref_body_pos_extend
        root_pos = torch.from_numpy(self.pos).float()

        
        # ref_body_pos_extend 没有问题
        diff_body_pos_extend = ref_body_pos_extend - self.rigid_body_pos_extend
        diff_local_body_pos_extend = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_body_pos_extend.view(-1, 3))
        self._obs_dif_local_rigid_body_pos = diff_local_body_pos_extend.view(1, -1)
        diff_local_body_pos_track = diff_local_body_pos_extend.view(ref_body_pos_extend.shape) - ref_body_pos_extend[:, 0:1, :]

        global_body_pos = ref_body_pos_extend - root_pos.unsqueeze(0)
        local_body_pos = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_body_pos.view(-1, 3))
        self._obs_local_ref_rigid_body_pos = local_body_pos.view(1, -1)

        global_ref_body_vel = ref_body_vel_extend.view(1, -1, 3)
        local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))
        self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(1, -1)

        ## diff compute - kinematic joint position
        self.dif_joint_angles = (ref_joint_pos - self.q).to(dtype=torch.float32).view(1, -1)
        ## diff compute - kinematic joint velocity
        self.dif_joint_velocities = (ref_joint_vel - self.dq).to(dtype=torch.float32).view(1, -1)      
        self._obs_global_ref_body_vel = global_ref_body_vel.view(1, -1) # (num_envs, num_rigid_bodies*3)
        self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(1, -1) # (num_envs, num_rigid_bodies*3)
        self._obs_local_ref_rigid_body_vel_relyaw = my_quat_rotate(relyaw_heading_inv_quat_expand.view(-1, 4), 
                                                                   global_ref_body_vel.view(-1, 3)).view(1, -1)
        
        if "future_ref_steps" in self.cfg.obs and self.cfg.obs.future_ref_steps > 0:
            future_motion_res = self.kick_motion_res_multiplestep()
            future_ref_joint_pos = future_motion_res["dof_pos"] # [num_envs, num_future_steps, num_dofs]
            future_ref_joint_vel = future_motion_res["dof_vel"] # [num_envs, num_future_steps, num_dofs]
            future_ref_body_pos_extend = future_motion_res["rg_pos_t"] # [num_envs, num_future_steps, num_markers, 3]
            
            self._obs_future_ref_dof_pos = future_ref_joint_pos.view(1, -1) # [num_envs, num_future_steps * num_dofs]
            self._obs_future_ref_dof_vel = future_ref_joint_vel.view(1, -1) # [num_envs, num_future_steps * num_dofs]
            future_ref_body_pos_extend = future_ref_body_pos_extend.view(1, -1, 3) # [num_envs, num_future_steps * num_markers, 3]
            future_global_ref_body_pos_extend = future_ref_body_pos_extend - root_pos.unsqueeze(0).unsqueeze(0)

            if "future_ref_steps" in self.cfg.obs and self.cfg.obs.future_ref_steps > 0:
                heading_inv_future = heading_inv_rot.unsqueeze(1).expand(-1, self.cfg.obs.future_ref_steps * self.num_rigid_bodies_extend, -1).reshape(-1, 4)
                future_local_ref_body_pos_flat = my_quat_rotate(heading_inv_future.view(-1, 4), future_global_ref_body_pos_extend.view(-1, 3))
                self._obs_future_local_ref_body_pos_extend = future_local_ref_body_pos_flat.view(1, -1) # (num_envs, num_rigid_bodies*3)

                future_dif_global_body_pos = future_ref_body_pos_extend.view(1, self.cfg.obs.future_ref_steps, -1, 3) - self.rigid_body_pos_extend.view(1, 1, -1, 3)
                heading_inv_future = heading_inv_rot.unsqueeze(1).expand(-1, self.cfg.obs.future_ref_steps * self.num_rigid_bodies_extend, -1).reshape(-1, 4)
                future_dif_local_body_pos = my_quat_rotate(heading_inv_future.view(-1, 4), future_dif_global_body_pos.view(-1, 3))
                self._obs_future_dif_local_rigid_body_pos = future_dif_local_body_pos.view(1, -1) # (num_envs, num_rigid_bodies*3)
            
        
        if self.counter % 5 == 0:
            # 累积数据到列表
            self.track_ref_accumulated['ref_dof_pos'].append(np.array(ref_joint_pos))
            self.track_ref_accumulated['ref_body_pos_extend'].append(np.array(ref_body_pos_extend))
            self.track_ref_accumulated['dof_pos'].append(np.array(self.q))
            self.track_ref_accumulated['body_pos_extend'].append(np.array(self.rigid_body_pos_extend))
            self.track_ref_accumulated['dif_local_rigid_body_pos'].append(np.array(diff_local_body_pos_track))
            self.track_ref_accumulated['counter'].append(int(self.counter / 5))
        self.counter += 1
        return

    # def save_batch(self, batch_num=None):
    #     """保存当前批次数据"""
    #     if batch_num is None:
    #         batch_num = len([f for f in os.listdir(self.save_dir) if f.startswith("batch_")])
            
    #     filename = os.path.join(self.save_dir, f"batch_{batch_num:04d}.npz")
        
    #     save_data = {}
    #     for key, value_list in self.track_ref_accumulated.items():
    #         if value_list:
    #             save_data[key] = np.array(value_list)
        
    #     np.savez(filename, **save_data)
    #     print(f"已保存批次 {batch_num} 到: {filename}")
        
    #     # 清空累积列表
    #     for key in self.track_ref_accumulated:
    #         self.track_ref_accumulated[key].clear()

    def UpdateObs(self):
        self.GetState()
        self.KickMotionLib()
        self.UpdateObsWoHistory()
        self.UpdateObsForHistory()
        
    def _check_init(self):
        assert self.dt is not None, "dt is not set"
        assert self.dt>0 and self.dt < 0.1, "dt is not in the valid range"
        assert self.cfg is not None or not isinstance(self.cfg, OmegaConf), "cfg is not set"
        
        assert self.num_dofs is not None, "num_dofs is not set"
        assert self.num_dofs == 23, "In policy level, only 23 dofs are supported for now"
        assert self.kp is not None and type(self.kp) == np.ndarray and self.kp.shape == (self.num_dofs,), "kp is not set"
        assert self.kd is not None and type(self.kd) == np.ndarray and self.kd.shape == (self.num_dofs,), "kd is not set"
        
        assert (self.dof_init_pose is not None and type(self.dof_init_pose) == np.ndarray and 
                    self.dof_init_pose.shape == (self.num_dofs,)), "dof_init_pose is not set"
        
        assert self.tau_limit is not None and type(self.tau_limit) == np.ndarray and self.tau_limit.shape == (self.num_dofs,), "tau_limit is not set"
        
        assert self.BYPASS_ACT is not None, "BYPASS_ACT is not set"
        assert self.BYPASS_ACT in [True, False], "BYPASS_ACT is not a boolean, got {self.BYPASS_ACT}"
        
        assert self._pid_size > 0, "pid_size is not correctly set"
    
    def _make_init_pose(self):
        cfg_init_state = self.cfg.robot.init_state
        self.body_names = self.cfg.robot.body_names
        self.dof_names = self.cfg.robot.dof_names
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        assert self.num_dofs == 23, "Only 23 dofs are supported for now"
        
        
        dof_init_pose = cfg_init_state.default_joint_angles
        dof_effort_limit_list = self.cfg.robot.dof_effort_limit_list
        
        self.dof_init_pose = np.array([dof_init_pose[name] for name in self.dof_names])
        self.tau_limit = np.array(dof_effort_limit_list)
        
        
        self.kp = np.zeros(self.num_dofs)
        self.kd = np.zeros(self.num_dofs)
        
        
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.robot.control.stiffness.keys():
                if dof_name in name:
                    self.kp[i] = self.cfg.robot.control.stiffness[dof_name]
                    self.kd[i] = self.cfg.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.kp[i]} and {self.kd[i]}")
            if not found:
                raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        
        motion_lib = self.motion_libs[0]
        if motion_lib is not None:
            # 获取第0帧的状态
            motion_times = torch.tensor(0.0, dtype=torch.float32)
            self.motion_ids = torch.zeros((1), dtype=torch.int32)
            motion_res = motion_lib.get_motion_state(self.motion_ids, motion_times)
        if USE_MOTION_LIB_INIT:
            # 从motion_lib的第0帧读取初始状态
            
            if motion_lib is not None:
                # 存储初始状态
                self.root_init_pos = motion_res["root_pos"].cpu().numpy().flatten()
                self.root_init_pos[2] += 0.05
                self.root_init_rot = motion_res["root_rot"].cpu().numpy().flatten()
                self.dof_init_pos = motion_res["dof_pos"].cpu().numpy().flatten()
                self.dof_init_vel = motion_res["dof_vel"].cpu().numpy().flatten()
                self.root_init_vel = motion_res["root_vel"].cpu().numpy().flatten()
                self.root_init_ang_vel = motion_res["root_ang_vel"].cpu().numpy().flatten()
                    
                    
        
    def _make_buffer(self):
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)
        
        self.q = np.zeros(self.num_dofs)
        self.dq = np.zeros(self.num_dofs)
        self.quat = np.zeros(4)  # XYZW
        self.omega = np.zeros(3)
        self.gvec = np.zeros(3)
        self.rpy = np.zeros(3)
        
        self.act = np.zeros(self.num_dofs)
        
        self.only_actor_obs_auxiliary = {'history_actor' : self.cfg.obs.obs_auxiliary['history_actor']}
        self.history_handler = HistoryHandler(1, self.only_actor_obs_auxiliary, self.cfg.obs.obs_dims, self.device)
        
        self.motion_lib = None
        self.ref_init_yaw = np.zeros(1,dtype=np.float32)
        self.relyaw = np.zeros(1,dtype=np.float32)
        self.dif_joint_angles = torch.zeros(self.num_dofs, dtype=torch.float32)
        self.dif_joint_velocities = torch.zeros(self.num_dofs, dtype=torch.float32)
        n_re = self.num_rigid_bodies_extend
        self._obs_global_ref_body_vel = torch.zeros(n_re * 3, dtype=torch.float32)
        self._obs_local_ref_rigid_body_vel = torch.zeros(n_re * 3, dtype=torch.float32)
        self._obs_future_dif_local_rigid_body_pos = torch.zeros(n_re * 3 * self.cfg.obs.future_ref_steps, dtype=torch.float32)
        self._obs_future_local_ref_body_pos_extend = torch.zeros(n_re * 3 * self.cfg.obs.future_ref_steps, dtype=torch.float32)

        self._obs_local_ref_rigid_body_pos_relyaw = torch.zeros(n_re * 3, dtype=torch.float32)
        
        # Initialize observation variables to prevent AttributeError during Reset
        self._obs_dif_local_rigid_body_pos = torch.zeros(n_re * 3, dtype=torch.float32)
        self._obs_local_ref_rigid_body_pos = torch.zeros(n_re * 3, dtype=torch.float32)
        self._obs_local_ref_rigid_body_vel_relyaw = torch.zeros(n_re * 3, dtype=torch.float32)
                
                
        ...
        
    def _make_motionlib_from_config(self):
        """从配置中创建motion_libs，用于初始化阶段"""
        self.motion_len = self.cfg.obs.motion_len
        
        # 创建motion_libs列表
        self.motion_libs: List[Optional[MotionLibRobot]] = []
        
        # 从配置中获取motion_file信息
        if hasattr(self.cfg.robot.motion, 'motion_file') and self.cfg.robot.motion.motion_file:
            m_cfg = DictConfig({
                'motion_file': self.cfg.robot.motion.motion_file,
                'asset': self.cfg.robot.motion.asset,
                'extend_config': self.cfg.robot.motion.extend_config,
            })
            
            motion_lib = MotionLibRobot(m_cfg, num_envs=1, device='cpu')
            # motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
            self.motion_libs.append(motion_lib)
            logger.info(f"Created motion_lib from config: {self.cfg.robot.motion.motion_file}")
        else:
            logger.warning("No motion_file found in config, motion_libs will be empty")
        
        logger.info(f"len(self.motion_libs): {len(self.motion_libs)}")
        
    def _make_motionlib(self, cfg_policies: List[URCIPolicyObs]):
        self.motion_len = self.cfg.obs.motion_len # an initial value
        
        # For all motion tracking policy, load the motion lib file
        self.motion_libs: List[Optional[MotionLibRobot]] = []
        
        for cfg_policy in cfg_policies:
            obs_cfg, policy_fn = cfg_policy 
            if isinstance(obs_cfg, DictConfig):
                m_cfg = DictConfig({
                    'motion_file': obs_cfg.motion_file,
                    'asset': self.cfg.robot.motion.asset,
                    'extend_config': self.cfg.robot.motion.extend_config,
                })
                
                motion_lib = MotionLibRobot(m_cfg, num_envs=1, device='cpu')
                self.num_motions = motion_lib._num_unique_motions
                logger.info(f"DEBUG: num_motions = {self.num_motions}")
                motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
                self.motion_libs.append(motion_lib)
            elif isinstance(obs_cfg, Callable):
                # self.motion_len = -1
                # pass
                self.motion_libs.append(None)
            else:
                raise ValueError(f"Invalid obs_cfg: {obs_cfg}")
            
            
        assert len(self.motion_libs) == len(cfg_policies), f"{len(self.motion_libs)=} != {len(cfg_policies)=}"
        logger.info(f"len(self.motion_libs): {len(self.motion_libs)}")
        # breakpoint()
        
        
    # ---- Save Motion System----
        
    # def _make_init_save_motion(self):
    #     save_dir = self.cfg.checkpoint.parent.parent / "motions"
    #     # os.makedirs(save_dir, exist_ok = True)

    #     self._save_motion_id = 0
        
    #     if hasattr(self.cfg, 'dump_motion_name'):
    #         raise NotImplementedError
    #         self.save_motion_dir = save_dir / (str(self.cfg.eval_timestamp) + "_" + self.cfg.dump_motion_name)
    #     else:
    #         self.save_motion_dir = save_dir / f"{self.cfg.env.config.save_note}_URCI_{type(self).__name__}_{self.cfg.eval_timestamp}"
    #     self.save_motion_dir.mkdir(parents=True, exist_ok=True)
    #     logger.info("Save Motion Dir: ", self.save_motion_dir)
    #     OmegaConf.save(self.cfg, self.save_motion_dir / "config.yaml")
        

    #     self._dof_axis = np.load('humanoidverse/utils/motion_lib/dof_axis.npy', allow_pickle=True)
    #     self._dof_axis = self._dof_axis.astype(np.float32)

    #     self.num_augment_joint = len(self.cfg.robot.motion.extend_config)
    #     self.motions_for_saving: Dict[str, List[np.ndarray]] = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'action':[], 'terminate':[],
    #                                 'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
    #     self.motion_times_buf = []

    #     # breakpoint()

    # def _get_motion_to_save(self)->Tuple[float, Dict[str, np.ndarray]]:
    #     raise NotImplementedError("Not implemented")
    
    # def TrySaveMotionStep(self):
    #     if self.save_motion:
    #         motion_time, motion_data = self._get_motion_to_save()
    #         self.motion_times_buf.append(motion_time)
    #         for key, value in motion_data.items():
    #             if not key in self.motions_for_saving.keys():
    #                 self.motions_for_saving[key] = []
    #                 assert isinstance(value, np.ndarray), f"key: {key} is not a np.ndarray"
    #             self.motions_for_saving[key].append(value)
    
    # def TrySaveMotionFile(self, pid: int):
    #     if self.save_motion:
    #         import joblib
    #         from termcolor import colored
    #         if self.motions_for_saving['root_trans_offset'] == []:
    #             return
            
    #         for k, v in self.motions_for_saving.items():
    #             self.motions_for_saving[k] = np.stack(v).astype(np.float32) # type: ignore
            
    #         self.motions_for_saving['motion_times'] = np.array(self.motion_times_buf, dtype=np.float32) # type: ignore
            
    #         dump_data = {}
    #         keys_to_save = self.motions_for_saving.keys()

    #         motion_key = f"motion{self._save_motion_id}"
    #         dump_data[motion_key] = {
    #             key: self.motions_for_saving[key] for key in keys_to_save
    #         }
    #         dump_data[motion_key]['fps'] = 1 / self.dt

    #         save_path = f'{self.save_motion_dir}/{self._save_motion_id}_pid{pid}_frame{len(self.motions_for_saving["dof"])}_{time.strftime("%Y%m%d_%H%M%S")}.pkl'
    #         joblib.dump(dump_data, save_path)
            
    #         logger.info(colored(f"Saved motion data to {save_path}", 'green'))

    #         self._save_motion_id += 1            
    #         self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'action':[], 'terminate':[],
    #                                     'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
    #         self.motion_times_buf = []
    #     ...

    def _update_init_state(self):
        if USE_MOTION_LIB_INIT:
            motion_times = torch.tensor(0.0, dtype=torch.float32)
            motion_res = self.motion_lib.get_motion_state(self.motion_ids, motion_times)
            
            # 存储初始状态
            self.root_init_pos = motion_res["root_pos"].cpu().numpy().flatten()
            self.root_init_pos[2] +=0.05
            self.root_init_rot = motion_res["root_rot"].cpu().numpy().flatten()
            self.dof_init_pos = motion_res["dof_pos"].cpu().numpy().flatten()
            self.dof_init_vel = motion_res["dof_vel"].cpu().numpy().flatten()
            self.root_init_vel = motion_res["root_vel"].cpu().numpy().flatten()
            self.root_init_ang_vel = motion_res["root_ang_vel"].cpu().numpy().flatten()
    # --------------------------------------------
    
    
    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return np2torch(self.cmd[:2]).view(1, -1)
    
    def _get_obs_command_ang_vel(self):
        return np2torch(self.cmd[2:3]).view(1, -1)
    
    def _get_obs_actions(self,):
        return np2torch(self.act).view(1, -1)
    
    def _get_obs_base_pos_z(self,):
        # raise NotImplementedError("Not Implemented")
        return np2torch(self.pos[2:3]).view(1, -1)
    
    def _get_obs_feet_contact_force(self,):
        raise NotImplementedError("Not implemented")
        return self.data.contact.force[:, :].view(self.num_envs, -1)
          
    def _get_obs_base_lin_vel(self,):
        return np2torch(self.vel).view(1, -1)
    
    def _get_obs_base_ang_vel(self,):
        return np2torch(self.omega).view(1, -1)
    
    def _get_obs_projected_gravity(self,):
        return np2torch(self.gvec).view(1, -1)
    
    def _get_obs_dof_pos(self,):
        return np2torch(self.q - self.dof_init_pose).view(1, -1)
    
    def _get_obs_dof_vel(self,):
        # print(f"dof_vel: mean:{self.dq.mean()}, std:{self.dq.std()}")
        return np2torch(self.dq).view(1, -1)
    
    def _get_obs_base_quat(self,):
        return np2torch(self.quat).view(1, -1)
    
    def _get_obs_base_ang_vel_noise(self,):
        return np2torch(self.omega).view(1, -1)
    
    def _get_obs_projected_gravity_noise(self,):
        return np2torch(self.gvec).view(1, -1)
    
    def _get_obs_dof_pos_noise(self,):
        return np2torch(self.q - self.dof_init_pose).view(1, -1)

    def _get_obs_dof_vel_noise(self,):
        return np2torch(self.dq).view(1, -1)

    def _get_obs_ref_motion_phase(self):
        # logger.info(f"Phase: {self.ref_motion_phase} | {self.motion_len}")
        return torch.tensor(self.ref_motion_phase).reshape(1,)
    
    def _get_obs_relyaw(self):
        return np2torch(self.relyaw).view(1, -1)
    
    def _get_obs_dif_joint_angles(self):
        # print(f"dif_joint_angles: mean:{self.dif_joint_angles.mean()}, std:{self.dif_joint_angles.std()}")
        return self.dif_joint_angles.view(1, -1)

    def _get_obs_dif_joint_velocities(self):
        # print(f"dif_joint_velocities: mean:{self.dif_joint_velocities.mean()}, std:{self.dif_joint_velocities.std()}")
        return self.dif_joint_velocities.view(1, -1)
    
    def _get_obs_global_ref_rigid_body_vel(self):
        # print(f"global_ref_body_vel: mean:{self._obs_global_ref_body_vel.mean()}, std:{self._obs_global_ref_body_vel.std()}")
        return self._obs_global_ref_body_vel.view(1, -1)
    
    def _get_obs_local_ref_rigid_body_vel(self):
        # print(f"local_ref_rigid_body_vel: mean:{self._obs_local_ref_rigid_body_vel.mean()}, std:{self._obs_local_ref_rigid_body_vel.std()}")
        return self._obs_local_ref_rigid_body_vel.view(1, -1)
    
    
    def _get_obs_local_ref_rigid_body_vel_relyaw(self):
        return self._obs_local_ref_rigid_body_vel_relyaw.view(1, -1)
    
    def _get_obs_dif_local_rigid_body_pos(self):
        return self._obs_dif_local_rigid_body_pos.view(1, -1)
    
    def _get_obs_local_ref_rigid_body_pos(self):
        return self._obs_local_ref_rigid_body_pos.view(1, -1)
    
    def _get_obs_future_local_ref_body_pos_extend(self):
        if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
            return self._obs_future_local_ref_body_pos_extend.view(1, self.cfg.obs.future_ref_steps, -1)
        else:
            return self._obs_future_local_ref_body_pos_extend.view(1, -1)
    
    def _get_obs_future_dif_local_rigid_body_pos(self):
        if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
            return self._obs_future_dif_local_rigid_body_pos.view(1, self.cfg.obs.future_ref_steps, -1)
        else:
            return self._obs_future_dif_local_rigid_body_pos.view(1, -1)
    
    def _get_obs_history_dif_body_pos(self):
        return self.rigid_body_pos_extend.clone().reshape(1, -1)
    
    def _get_obs_history_body_pos(self):
        return self.rigid_body_pos_extend.clone().reshape(1, -1)
    
    
    def _get_obs_history_actor(self,):
        # obs_cfg_obs = self._obs_cfg_obs
        # assert "history_actor" in obs_cfg_obs.obs_auxiliary.keys()
        # history_config = obs_cfg_obs.obs_auxiliary['history_actor']
        # history_key_list = history_config.keys()
        # history_tensors = []
        # for key in sorted(history_config.keys()):
        #     history_length = history_config[key]
        #     history_tensor = self.history_handler.query(key)[:, :history_length]
        #     history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
        #     history_tensors.append(history_tensor)
            
        # # print("history_tensors:", {key: history_tensors[i].shape for i, key in enumerate(sorted(history_config.keys()))})
        # # breakpoint()
        # return torch.cat(history_tensors, dim=1).reshape(-1)
        obs_cfg_obs = self._obs_cfg_obs
        assert "history_actor" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        # history_tensors = []
        if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
            transformer_history=[]
        else:
            history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]

            if key == "history_dif_body_pos":
                num_rigid_bodies = self.num_rigid_bodies_extend
                history_body_pos = history_tensor.view(1, history_length, -1, 3)
                history_body_pos_diff = history_body_pos - self.rigid_body_pos_extend.view(1, 1, -1, 3) 
                heading_inv_rot = calc_heading_quat_inv(torch.from_numpy(self.quat).float().unsqueeze(0), w_last=True) #xyzw
                heading_inv_history = heading_inv_rot.unsqueeze(1).expand(-1, history_length * self.num_rigid_bodies_extend, -1).reshape(-1, 4)
                history_tensor = my_quat_rotate(heading_inv_history.view(-1, 4), history_body_pos_diff.view(-1, 3)).view(1, history_length, -1)
            if key == "history_body_pos":
                num_rigid_bodies = self.num_rigid_bodies_extend
                history_body_pos = history_tensor.view(1, -1, 3)
                history_body_pos_diff = history_body_pos - torch.from_numpy(self.pos).float().view(1, 1, 3)
                heading_inv_rot = calc_heading_quat_inv(torch.from_numpy(self.quat).float().unsqueeze(0), w_last=True) #xyzw
                heading_inv_history = heading_inv_rot.unsqueeze(1).expand(-1, history_length * self.num_rigid_bodies_extend, -1).reshape(-1, 4)
                history_tensor = my_quat_rotate(heading_inv_history.view(-1, 4), history_body_pos_diff.view(-1, 3)).view(1, history_length, -1)

            if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
                history_tensor_transformer = history_tensor.clone()
                transformer_history.append(history_tensor_transformer)
            else:
                history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
                history_tensors.append(history_tensor)
        if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
            return torch.cat(transformer_history, dim=2)
        else:
            return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self,):
        # obs_cfg_obs = self._obs_cfg_obs
        # assert "history_critic" in obs_cfg_obs.obs_auxiliary.keys()
        # history_config = obs_cfg_obs.obs_auxiliary['history_critic']
        # history_key_list = history_config.keys()
        # history_tensors = []
        # for key in sorted(history_config.keys()):
        #     history_length = history_config[key]
        #     history_tensor = self.history_handler.query(key)[:, :history_length]
        #     history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
        #     history_tensors.append(history_tensor)
        # return torch.cat(history_tensors, dim=1).reshape(-1)
        obs_cfg_obs = self._obs_cfg_obs
        assert "history_critic" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        # history_tensors = []
        if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
            transformer_history=[]
        else:
            history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]

            if key == "history_dif_body_pos":
                num_rigid_bodies = self.num_rigid_bodies_extend
                history_body_pos = history_tensor.view(1, history_length, -1, 3)
                history_body_pos_diff = history_body_pos - self.rigid_body_pos_extend.view(1, 1, -1, 3) 
                heading_inv_rot = calc_heading_quat_inv(torch.from_numpy(self.quat).float().unsqueeze(0), w_last=True) #xyzw
                heading_inv_history = heading_inv_rot.unsqueeze(1).expand(-1, history_length * self.num_rigid_bodies_extend, -1).reshape(-1, 4)
                history_tensor = my_quat_rotate(heading_inv_history.view(-1, 4), history_body_pos_diff.view(-1, 3)).view(1, history_length, -1)
            if key == "history_body_pos":
                num_rigid_bodies = self.num_rigid_bodies_extend
                history_body_pos = history_tensor.view(1, -1, 3)
                history_body_pos_diff = history_body_pos - torch.from_numpy(self.pos).float().view(1, 1, 3)
                heading_inv_rot = calc_heading_quat_inv(torch.from_numpy(self.quat).float().unsqueeze(0), w_last=True) #xyzw
                heading_inv_history = heading_inv_rot.unsqueeze(1).expand(-1, history_length * self.num_rigid_bodies_extend, -1).reshape(-1, 4)
                history_tensor = my_quat_rotate(heading_inv_history.view(-1, 4), history_body_pos_diff.view(-1, 3)).view(1, history_length, -1)
            if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
                history_tensor_transformer = history_tensor.clone()
                transformer_history.append(history_tensor_transformer)
            else:
                history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
                history_tensors.append(history_tensor)
        if 'use_transformer' in self.cfg.env.config and self.cfg.env.config.use_transformer:
            return torch.cat(transformer_history, dim=2)
        else:
            return torch.cat(history_tensors, dim=1)