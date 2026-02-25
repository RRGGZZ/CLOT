import torch
import numpy as np
from pathlib import Path
import os
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.utils.torch_utils import *
from isaac_utils.rotations import (
    slerp,
    my_quat_rotate,
    calc_heading_quat,
    calc_heading_quat_inv,
    quat_mul,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
    get_euler_xyz_in_tensor,
    calc_yaw_heading_quat_inv,
    apply_rotation_to_quat_z
)
# from isaacgym import gymtorch, gymapi, gymutil
from scipy.spatial.transform import Rotation as sRot

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from humanoidverse.utils.lpf import ActionFilterButterTorch

from termcolor import colored
from loguru import logger

import joblib

from typing import Dict, Optional

class WholeBodyState:
    # root_pos: torch.Tensor
    # root_vel: torch.Tensor
    # root_quat: torch.Tensor # XYZW
    # root_ang_vel: torch.Tensor
    root_state: torch.Tensor
    # dof_state: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    
    def __init__(self, root_state, dof_pos, dof_vel):
        self.root_state = root_state
        self.dof_pos = dof_pos
        self.dof_vel = dof_vel

    def interpolate(self, other, alpha):
        # alpha in [0, 1]
        # result = self * (1- alpha) + other * (alpha)
        # result is a WholeBodyState object
        result = WholeBodyState(
            root_state=self.root_state * (1- alpha) + other.root_state * (alpha),
            dof_pos=self.dof_pos * (1- alpha) + other.dof_pos * (alpha),
            dof_vel=self.dof_vel * (1- alpha) + other.dof_vel * (alpha)
        )
        
        
        result.root_state[:, 3:7] = slerp(self.root_state[:, 3:7], other.root_state[:, 3:7], 
                                            torch.tensor([alpha], device=self.root_state.device))
        
        
        return result

    def apply(self, simulator):
        assert not torch.isnan(self.root_state).any(), "Root state contains NaN values"
        assert not torch.isnan(self.dof_pos).any(), "DOF position contains NaN values"
        assert not torch.isnan(self.dof_vel).any(), "DOF velocity contains NaN values"
        
        num_envs, device = self.root_state.shape[0], self.root_state.device
        
        simulator.robot_root_states[:, :] = self.root_state[:, :]
        simulator.dof_pos[:, :] = self.dof_pos[:, :]
        simulator.dof_vel[:, :] = self.dof_vel[:, :]
        
        refresh_env_ids = torch.arange(num_envs, device=device)
        simulator.set_actor_root_state_tensor(refresh_env_ids, simulator.all_root_states)
        simulator.set_dof_state_tensor(refresh_env_ids, simulator.dof_state)

def radial_velocity_potential(cur_vel, ref_vel, sigma_theta=0.75, alpha_norm=0.4):
    # assume cur_vel, ref_vel: [N, d]
    cosine_similarity = torch.nn.functional.cosine_similarity(cur_vel, ref_vel, dim=-1)
    cosine_error = 1-cosine_similarity
    potential_cosine = torch.exp(-cosine_error / sigma_theta)
    # print(f'{potential_cosine=}','\t|',f'{cosine_error=}')
    
    
    norm_cur_velocity = cur_vel.norm(dim=-1)
    norm_ref_velocity = ref_vel.norm(dim=-1)
    norm_ratio = norm_cur_velocity / norm_ref_velocity
    
    potential_norm = norm_ratio*torch.exp(alpha_norm*(1-norm_ratio**(1/alpha_norm)))
    
    
    potential_velocity = potential_cosine * potential_norm
    return potential_velocity


class LeggedRobotMotionTracking(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        self.debug_viz = True
        # import ipdb; ipdb.set_trace()
        super().__init__(config, device)
        self._init_motion_lib()
        self._init_motion_extend()
        self._init_tracking_config()
        # self._expand_sigma()

        self.init_done = True
        self.debug_viz = True

        self._init_save_motion()

        if self.config.use_teleop_control:
            self.teleop_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            self.node = Node("motion_tracking")
            self.teleop_sub = self.node.create_subscription(Float64MultiArray, "vision_pro_data", self.teleop_callback, 1)

        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        else:
            self.terminate_when_motion_far_threshold = self.config.termination_scales.termination_motion_far_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        if self.config.termination_scales.termination_scale and self.config.termination_curriculum.termination_scale_curriculum:
            self.termination_scale_value = self.config.termination_curriculum.termination_scale_initial_value
            logger.info(f"Terminate scale value: {self.termination_scale_value}")
        if self.config.termination.terminate_when_dof_far:
            self.terminate_when_dof_far_threshold = self.config.termination_curriculum.terminate_when_dof_far_curriculum.init
            logger.info(f"Terminate when dof far threshold: {self.terminate_when_dof_far_threshold}")


        if 'enable' in self.config.soft_dynamic_correction and self.config.soft_dynamic_correction.enable:
            self.sdc_alpha = self.config.soft_dynamic_correction.alpha
            logger.info(f"Soft Dynamic Correction alpha: {self.sdc_alpha}")
        
        if "action_filt" in self.config.robot.control and self.config.robot.control.action_filt:
            self.action_filter = ActionFilterButterTorch(lowcut=np.zeros(self.num_envs*self.dim_actions),
                                                        highcut=np.ones(self.num_envs*self.dim_actions) * self.config.robot.control.action_cutfreq, 
                                                        sampling_rate=1./self.dt, num_joints=self.num_envs * self.dim_actions, 
                                                        device=self.device)

    def teleop_callback(self, msg):
        self.teleop_marker_coords = torch.tensor(msg.data, device=self.device)

    def _init_save_motion(self):
        if "save_motion" in self.config:
            print(f"DEBUG: {self.motion_len[0]=}")
            print(f"DEBUG: {self.dt=}")
            print(f"DEBUG: {self.motion_len[0]/self.dt=}")
            self._motion_episode_length = torch.ceil((self.motion_len/self.dt)[0]).int()
            assert (self._motion_episode_length- self.motion_len[0]/self.dt).norm() <1, f"Motion length {self.motion_len} is not divisible by motion dt {self.dt}, {self._motion_episode_length=}, {(self._motion_episode_length- self.motion_len[0]/self.dt).norm()=}"
            assert self._motion_episode_length > 0, f"Motion length {self.motion_len} is not positive"
            
            self.save_motion = self.config.save_motion
            if self.save_motion:
                os.makedirs(Path(self.config.ckpt_dir) / "motions", exist_ok = True)

                
                if hasattr(self.config, 'dump_motion_name'):
                    raise NotImplementedError
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / (str(self.config.eval_timestamp) + "_" + self.config.dump_motion_name)
                else:
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / f"{self.config.save_note}_{self.config.eval_timestamp}"
                self.save_motion = True
                self.num_augment_joint = len(self.config.robot.motion.extend_config)
                self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                            'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[],'contact_mask':[]}
                self.motion_times_buf = []
                self.start_save = False
                
                self._write_to_file = True

        else:
            self.save_motion = False

    def _init_motion_lib(self):
        self.config.robot.motion.step_dt = self.dt
        
        assert self.config.robot.motion.motion_lib_type in ['origin', 'WJX', 'ZTJ'], "motion_lib_type must be either 'origin' or 'WJX' or 'ZTJ'"
        if self.config.robot.motion.motion_lib_type == 'origin':
            from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
            self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        elif self.config.robot.motion.motion_lib_type == 'WJX':
            from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX
            self._motion_lib = MotionLibRobotWJX(self.config.robot.motion, num_envs=self.num_envs, device=self.device)

        elif self.config.robot.motion.motion_lib_type == 'ZTJ':
            from humanoidverse.utils.motion_lib.motion_lib_robot_ztj import MotionLibRobotZTJ
            self._motion_lib = MotionLibRobotZTJ(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        self.num_motions = self._motion_lib._num_unique_motions
        # if self.is_evaluating:
        #     self._motion_lib.load_motions(random_sample=False)
        # else:
        #     self._motion_lib.load_motions(random_sample=True)
        if self.is_evaluating:
            self.motion_ids = self._motion_lib.sample_motions(n=self.num_envs, random_sample=False)
            self.motion_count = torch.bincount(self.motion_ids.int(), minlength=self.num_motions)
        else:
            self.motion_ids = self._motion_lib.sample_motions(n=self.num_envs, random_sample=True)
            self.motion_count = torch.bincount(self.motion_ids.int(), minlength=self.num_motions)

        ref_init_state = self.kick_motion_res() ; self._kick_motion_res_counter = -1
        self.ref_init_rpy = get_euler_xyz_in_tensor(ref_init_state['root_rot'][:1]) # [1,3]
        # breakpoint()

        self._resample_motion_times(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt[self.motion_ids]
        self.motion_start_idx = 0
        self.adaptive_prob = torch.ones(self._motion_lib._num_unique_motions, device=self.device) / self._motion_lib._num_unique_motions
        
        
        if hasattr(self._motion_lib, 'has_contact_mask') and self._motion_lib.has_contact_mask:
            self.ref_contact_mask = torch.zeros(self.num_envs, self._motion_lib._contact_size, # type: ignore
                                                dtype=torch.float, device=self.device, requires_grad=False)
            
        # self.ref_init_yaw = 

    def _init_tracking_config(self):
        
        
        
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
            print(f"motion_tracking_id: {self.motion_tracking_id}")
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
        if self.config.resample_motion_when_training:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
            
        self.end_time_ratio_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

        self.feet_body_indices = torch.zeros(len(self.feet_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, feet_body_name in enumerate(self.feet_body_names):
            self.feet_body_indices[i] = self.simulator._body_list.index(feet_body_name)
        
    def _init_motion_extend(self):
        if "extend_config" in self.config.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                # extend_parent_ids.append(self.simulator.find_rigid_body_indice(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)

            self.marker_coords = torch.zeros(self.num_envs, 
                                         self.num_bodies + self.num_extend_bodies, 
                                         3, 
                                         dtype=torch.float, 
                                         device=self.device, 
                                         requires_grad=False) # extend
            
            self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.ref_body_vel_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def start_compute_metrics(self):
        self.compute_metrics = True
        self.start_idx = 0
    
    def forward_motion_samples(self):
        pass
    
    def _init_buffers(self):
        super()._init_buffers()
        self._init_adaptive_sigma()
        self.vr_3point_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        # self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.feet_air_min_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

        
    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _reset_buffers_callback(self, env_ids, target_buf=None):
        if len(env_ids) == 0:
            return
        super()._reset_buffers_callback(env_ids, target_buf=target_buf)
    
    def _reset_tasks_callback(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        
        
        end_time = self.last_episode_length_buf[env_ids]*self.dt
        self.end_time_ratio_buf[env_ids] = (end_time / (self.motion_len[env_ids] - self.motion_start_times[env_ids]).clamp(min=1e-4)).clamp(max=1.0)
        self.log_dict["end_time_ratio_std"] = self.end_time_ratio_buf.std()
        # breakpoint()
        
        self._update_average_episode_motion_end_ratio(env_ids)
        
        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._resample_motion_times(env_ids) # need to resample before reset root states
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self._update_terminate_when_motion_far_curriculum()
        
        if self.config.termination_scales.termination_scale and self.config.termination_curriculum.termination_scale_curriculum:
            self._update_termination_scale_curriculum()

        if self.config.termination.terminate_when_dof_far and self.config.termination_curriculum.terminate_when_dof_far_curriculum.enable:
            self._update_terminate_when_dof_far_curriculum()
            
        if 'enable' in self.config.soft_dynamic_correction and self.config.soft_dynamic_correction.enable and self.config.soft_dynamic_correction.curriculum.enable:
            self._update_soft_dynamic_correction()
    
    def _update_soft_dynamic_correction(self):
        sdc_curr = self.config.soft_dynamic_correction.curriculum
        
        if self.average_end_time_ratio < sdc_curr.level_down_threshold:
            self.sdc_alpha *= (1 + sdc_curr.degree)
        elif self.average_end_time_ratio > sdc_curr.level_up_threshold:
            self.sdc_alpha *= (1 - sdc_curr.degree)
        self.sdc_alpha = np.clip(self.sdc_alpha, sdc_curr.min_alpha, sdc_curr.max_alpha)

    def _update_terminate_when_dof_far_curriculum(self):
        dof_far_curr = self.config.termination_curriculum.terminate_when_dof_far_curriculum
        
        if self.average_end_time_ratio < dof_far_curr.level_down_threshold:
            self.terminate_when_dof_far_threshold *= (1 + dof_far_curr.degree)
        elif self.average_end_time_ratio > dof_far_curr.level_up_threshold:
            self.terminate_when_dof_far_threshold *= (1 - dof_far_curr.degree)
        if self.config.use_end_time_ratio:
            self.terminate_when_dof_far_threshold = float(self.average_end_time_ratio * dof_far_curr.min + (1 - self.average_end_time_ratio) * dof_far_curr.max)**4
        self.terminate_when_dof_far_threshold = np.clip(self.terminate_when_dof_far_threshold, 
                                                        dof_far_curr.min, 
                                                        dof_far_curr.max)
        
    def _update_obs_noise_curriculum(self):
        if self.average_end_time_ratio < self.config.obs.soft_dof_pos_curriculum_level_down_threshold:
            self.current_noise_curriculum_value *= (1 - self.config.obs.soft_dof_pos_curriculum_degree)
        elif self.average_end_time_ratio > self.config.rewards.reward_penalty_level_up_threshold:
            self.current_noise_curriculum_value *= (1 + self.config.obs.soft_dof_pos_curriculum_degree)
        if self.config.use_end_time_ratio:
            self.current_noise_curriculum_value = float(self.average_end_time_ratio * self.config.obs.noise_value_max + (1 - self.average_end_time_ratio) * self.config.obs.noise_value_min)**4
        self.current_noise_curriculum_value = np.clip(self.current_noise_curriculum_value, 
                                                      self.config.obs.noise_value_min, 
                                                      self.config.obs.noise_value_max)
        
    def _update_reward_penalty_curriculum(self):
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        if self.average_end_time_ratio < self.config.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.config.rewards.reward_penalty_degree)
        elif self.average_end_time_ratio > self.config.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.config.rewards.reward_penalty_degree)

        if self.config.use_end_time_ratio:
            self.reward_penalty_scale = float(self.average_end_time_ratio * self.config.rewards.reward_max_penalty_scale + (1 - self.average_end_time_ratio) * self.config.rewards.reward_min_penalty_scale)**4
        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.config.rewards.reward_min_penalty_scale, self.config.rewards.reward_max_penalty_scale)
    
    def _update_born_offset_curriculum(self):
        if self.average_end_time_ratio < self.config.domain_rand.born_offset_level_down_threshold:
            self.born_distance *= (1 - self.config.domain_rand.level_degree)
        elif self.average_end_time_ratio > self.config.domain_rand.born_offset_level_up_threshold:
            self.born_distance *= (1 + self.config.domain_rand.level_degree)
        # import ipdb; ipdb.set_trace()
        # torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
        if self.config.use_end_time_ratio:
            self.born_distance = float(self.average_end_time_ratio * self.config.domain_rand.born_offset_range[1] + (1 - self.average_end_time_ratio) * self.config.domain_rand.born_offset_range[0])**4
        self.born_distance = np.clip(self.born_distance, self.config.domain_rand.born_offset_range[0], self.config.domain_rand.born_offset_range[1]).item()
    
    def _update_born_heading_curriculum(self):
        if self.average_end_time_ratio < self.config.domain_rand.born_heading_level_down_threshold:
            self.born_heading_degree *= (1 - self.config.domain_rand.born_heading_level_degree)
        elif self.average_end_time_ratio > self.config.domain_rand.born_heading_level_up_threshold:
            self.born_heading_degree *= (1 + self.config.domain_rand.born_heading_level_degree)
        # import ipdb; ipdb.set_trace()
        # torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
        if self.config.use_end_time_ratio:
            self.born_heading_degree = float(self.average_end_time_ratio * self.config.domain_rand.born_heading_range[1] + (1 - self.average_end_time_ratio) * self.config.domain_rand.born_heading_range[0])**4
        self.born_heading_degree = np.clip(self.born_heading_degree, self.config.domain_rand.born_heading_range[0], self.config.domain_rand.born_heading_range[1])
    
    def _update_push_robots_interval_curriculum(self):
        if self.average_end_time_ratio < self.config.domain_rand.push_robots_level_down_threshold:
            self.push_robots_interval *= (1 + self.config.domain_rand.push_robots_level_degree)
        elif self.average_end_time_ratio > self.config.domain_rand.push_robots_level_up_threshold:
            self.push_robots_interval *= (1 - self.config.domain_rand.push_robots_level_degree)
        # import ipdb; ipdb.set_trace()
        # torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
        if self.config.use_end_time_ratio:
            self.push_robots_interval = float(self.average_end_time_ratio * self.config.domain_rand.push_robots_range[0] + (1 - self.average_end_time_ratio) * self.config.domain_rand.push_robots_range[1])**4
        self.push_robots_interval = np.clip(self.push_robots_interval, self.config.domain_rand.push_robots_range[0], self.config.domain_rand.push_robots_range[1])
    
    def _update_appaly_advance_curriculum(self):
        if self.average_end_time_ratio < self.config.domain_rand.advance_level_down_threshold:
            self.appaly_advance_interval *= (1 + self.config.domain_rand.advance_degree)
        elif self.average_end_time_ratio > self.config.domain_rand.advance_level_up_threshold:
            self.appaly_advance_interval *= (1 - self.config.domain_rand.advance_degree)
        if self.config.use_end_time_ratio:
            self.appaly_advance_interval = float(self.average_end_time_ratio * self.config.domain_rand.appaly_advance_range[0] + (1 - self.average_end_time_ratio) * self.config.domain_rand.appaly_advance_range[1])**4
        self.appaly_advance_interval = np.clip(self.appaly_advance_interval, self.config.domain_rand.appaly_advance_range[0], self.config.domain_rand.appaly_advance_range[1])
    
    def _update_motion_time_advance_curriculum(self):
        if self.average_end_time_ratio < self.config.domain_rand.advance_level_down_threshold:
            self.advance_initial_degree *= (1 - self.config.domain_rand.advance_degree)
        elif self.average_end_time_ratio > self.config.domain_rand.advance_level_up_threshold:
            self.advance_initial_degree *= (1 + self.config.domain_rand.advance_degree)
        if self.config.use_end_time_ratio:
            self.advance_initial_degree = float(self.average_end_time_ratio * self.config.domain_rand.advance_range[1] + (1 - self.average_end_time_ratio) * self.config.domain_rand.advance_range[0])**4
        self.advance_initial_degree = np.clip(self.advance_initial_degree, self.config.domain_rand.advance_range[0], self.config.domain_rand.advance_range[1]).item()
    
    def _update_reward_limits_curriculum(self):
        """
        Update the reward limits curriculum based on the average episode length.
        """
        curriculum_cfg = self.config.rewards.reward_limit.reward_limits_curriculum
        
        if self.use_reward_limits_dof_pos_curriculum:
            if self.average_end_time_ratio < curriculum_cfg.soft_dof_pos_curriculum_level_down_threshold:
                self.soft_dof_pos_curriculum_value *= (1 + curriculum_cfg.soft_dof_pos_curriculum_degree)
            elif self.average_end_time_ratio > curriculum_cfg.soft_dof_pos_curriculum_level_up_threshold:
                self.soft_dof_pos_curriculum_value *= (1 - curriculum_cfg.soft_dof_pos_curriculum_degree)
            if self.config.use_end_time_ratio:
                self.soft_dof_pos_curriculum_value = float(self.average_end_time_ratio * curriculum_cfg.soft_dof_pos_min_limit + (1 - self.average_end_time_ratio) * curriculum_cfg.soft_dof_pos_max_limit)**4
            self.soft_dof_pos_curriculum_value = np.clip(
                self.soft_dof_pos_curriculum_value,
                curriculum_cfg.soft_dof_pos_min_limit,
                curriculum_cfg.soft_dof_pos_max_limit
            )
        
        if self.use_reward_limits_dof_vel_curriculum:
            if self.average_end_time_ratio < curriculum_cfg.soft_dof_vel_curriculum_level_down_threshold:
                self.soft_dof_vel_curriculum_value *= (1 + curriculum_cfg.soft_dof_vel_curriculum_degree)
            elif self.average_end_time_ratio > curriculum_cfg.soft_dof_vel_curriculum_level_up_threshold:
                self.soft_dof_vel_curriculum_value *= (1 - curriculum_cfg.soft_dof_vel_curriculum_degree)
            if self.config.use_end_time_ratio:
                self.soft_dof_vel_curriculum_value = float(self.average_end_time_ratio * curriculum_cfg.soft_dof_vel_min_limit + (1 - self.average_end_time_ratio) * curriculum_cfg.soft_dof_vel_max_limit)**4
            self.soft_dof_vel_curriculum_value = np.clip(
                self.soft_dof_vel_curriculum_value,
                curriculum_cfg.soft_dof_vel_min_limit,
                curriculum_cfg.soft_dof_vel_max_limit
            )
        
        if self.use_reward_limits_torque_curriculum:
            if self.average_end_time_ratio < curriculum_cfg.soft_torque_curriculum_level_down_threshold:
                self.soft_torque_curriculum_value *= (1 + curriculum_cfg.soft_torque_curriculum_degree)
            elif self.average_end_time_ratio > curriculum_cfg.soft_torque_curriculum_level_up_threshold:
                self.soft_torque_curriculum_value *= (1 - curriculum_cfg.soft_torque_curriculum_degree)
            if self.config.use_end_time_ratio:
                self.soft_torque_curriculum_value = float(self.average_end_time_ratio * curriculum_cfg.soft_torque_min_limit + (1 - self.average_end_time_ratio) * curriculum_cfg.soft_torque_max_limit)**4
            self.soft_torque_curriculum_value = np.clip(
                self.soft_torque_curriculum_value,
                curriculum_cfg.soft_torque_min_limit,
                curriculum_cfg.soft_torque_max_limit
            )

    def _update_terminate_when_motion_far_curriculum(self):
        assert self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum
        if self.average_end_time_ratio < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.average_end_time_ratio > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        if self.config.use_end_time_ratio:
            self.terminate_when_motion_far_threshold = float(self.average_end_time_ratio * self.config.termination_curriculum.terminate_when_motion_far_threshold_min + (1 - self.average_end_time_ratio) * self.config.termination_curriculum.terminate_when_motion_far_threshold_max)**4
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)
    
    def _update_termination_scale_curriculum(self):
        assert self.config.termination_scales.termination_scale and self.config.termination_curriculum.termination_scale_curriculum
        if self.average_end_time_ratio < self.config.termination_curriculum.termination_scale_curriculum_level_down_threshold:
            self.termination_scale_value *= (1 - self.config.termination_curriculum.termination_scale_curriculum_degree)
        elif self.average_end_time_ratio > self.config.termination_curriculum.termination_scale_curriculum_level_up_threshold:
            self.termination_scale_value *= (1 + self.config.termination_curriculum.termination_scale_curriculum_degree)
        self.termination_scale_value = np.clip(self.termination_scale_value, 
                                                         self.config.termination_curriculum.termination_scale_value_min, 
                                                         self.config.termination_curriculum.termination_scale_value_max)    

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                # logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_task()

    def set_is_evaluating(self):
        super().set_is_evaluating()

    def _update_reset_buf(self):
        super()._update_reset_buf()
        
        # print(f"DEBUG: motion far = {torch.norm(self.dif_global_body_pos, dim=-1).max()}\t|  threshold={self.terminate_when_motion_far_threshold}")
        
        if self.config.termination.terminate_when_motion_far:
            reset_buf_motion_far = torch.any(torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
            self.reset_buf_terminate_by["motion_far"] = reset_buf_motion_far
            self.reset_buf |= reset_buf_motion_far
            # log current motion far threshold
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)
                
                
        if self.config.termination.terminate_when_dof_far:
            reset_buf_dof_far = torch.any(torch.norm(self.dif_joint_angles, dim=-1) > self.terminate_when_dof_far_threshold, dim=-1)
            self.reset_buf_terminate_by["dof_far"] = reset_buf_dof_far
            self.reset_buf |= reset_buf_dof_far
            # log current dof far threshold
            if self.config.termination_curriculum.terminate_when_dof_far_curriculum.enable:
                self.log_dict["terminate_when_dof_far_threshold"] = torch.tensor(self.terminate_when_dof_far_threshold, dtype=torch.float)

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.reset_buf_terminate_by["motion_end"] = current_time > self.motion_len
            self.track_motion_end_ratio += self.reset_buf_terminate_by["motion_end"]
            self.time_out_buf |= self.reset_buf_terminate_by["motion_end"]
    
    def _update_average_episode_motion_end_ratio(self, env_ids):
        num = len(env_ids)
        current_average_end_time_ratio = torch.mean(self.end_time_ratio_buf[env_ids], dtype=torch.float)

        self.average_end_time_ratio = self.average_end_time_ratio * (1 - num / self.num_compute_average_epl) + current_average_end_time_ratio * (num / self.num_compute_average_epl)
        self.log_dict["average_end_time_ratio"] = self.average_end_time_ratio


            
            

    def next_task(self):
        # This function is only called when evaluating
        self.motion_start_idx += self.num_envs
        logger.info(f"DEBUG: motion_start_idx = {self.motion_start_idx}")
        logger.info(f"DEBUG: self.num_motions = {self.num_motions}")
        if self.motion_start_idx >= self.num_motions:
            self.motion_start_idx = 0
        self.motion_ids = self._motion_lib.sample_motions(n=self.num_envs, start_idx=self.motion_start_idx, random_sample=False)
        self.reset_all()

    def resample_task(self):
        self.motion_start_idx = 0
        logger.info(f"DEBUG: motion_start_idx = {self.motion_start_idx}")
        logger.info(f"DEBUG: self.num_motions = {self.num_motions}")
        logger.info(f"Resampling motions for training at step {self.common_step_counter}")
        
        for i in range(self.num_motions):
            self.adaptive_prob[i] = self.track_motion_end_ratio[torch.nonzero(self.motion_ids.int() == i, as_tuple=False).flatten()].float().sum() \
                / (self.reset_buf_track[torch.nonzero(self.motion_ids.int() == i, as_tuple=False).flatten()].float().sum() + 1e-10)
        self.adaptive_prob = torch.clamp((1 - self.adaptive_prob) / (1 - self.adaptive_prob).sum(), min=1/(2*self.num_motions), max=2/self.num_motions)
        
        self.log_dict["adaptive_prob"] = self.adaptive_prob.mean()
        self.motion_ids = self._motion_lib.sample_motions(n=self.num_envs, random_sample=False, adaptive_sampling=True, adaptive_prob=self.adaptive_prob)
        logger.info(f"New motion ids: {self.motion_ids}")
        self.track_motion_end_ratio = torch.zeros_like(self.track_motion_end_ratio)
        self.reset_buf_track = torch.zeros_like(self.reset_buf_track)
        self.reset_all()

    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        self._kick_motion_res_counter = -1
        # self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
        # offset = self.env_origins
        # motion_times = (self.episode_length_buf ) * self.dt + self.motion_start_times # next frames so +1
        # # motion_res = self._get_state_from_motionlib_cache(self.motion_ids, motion_times, offset= offset)
        # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)

    def resample_motion(self):
        self._motion_lib.load_motions(random_sample=True)
        
        
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))


    def _compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            try:
                assert rew.shape[0] == self.num_envs
            except:
                import ipdb; ipdb.set_trace()
            # penalty curriculum
            if name in self.config.rewards.reward_penalty_reward_names:
                if self.config.rewards.reward_penalty_curriculum:
                    penalty_scale = self.reward_penalty_scale
                    if hasattr(self.config.rewards, "reward_ranges") and name in self.config.rewards.reward_ranges:
                        min_val, max_val = self.config.rewards.reward_ranges[name]
                        penalty_scale = np.clip(penalty_scale, min_val, max_val)
                        self.log_dict[f"{name}_scale"] = torch.tensor(penalty_scale, dtype=torch.float)
                    rew *= penalty_scale
            if self.config.use_vec_reward:
                self.rew_buf[:,i] += rew
            else:
                self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.config.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"] * self.termination_scale_value

            if self.config.use_vec_reward:
                self.rew_buf[:,i+1] += rew
            else:
                self.rew_buf += rew            
            self.episode_sums["termination"] += rew
            self.log_dict["termination_scale"] = torch.tensor(self.termination_scale_value, dtype=torch.float)
        
        if self.use_reward_penalty_curriculum:
            self.log_dict["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float)
            self.log_dict["average_episode_length"] = self.average_episode_length
        
        if self.use_reward_limits_dof_pos_curriculum:
            self.log_dict["soft_dof_pos_curriculum_value"] = torch.tensor(self.soft_dof_pos_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_dof_vel_curriculum:
            self.log_dict["soft_dof_vel_curriculum_value"] = torch.tensor(self.soft_dof_vel_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_torque_curriculum:
            self.log_dict["soft_torque_curriculum_value"] = torch.tensor(self.soft_torque_curriculum_value, dtype=torch.float)
        
        if self.add_noise_currculum:
            self.log_dict["current_noise_curriculum_value"] = torch.tensor(self.current_noise_curriculum_value, dtype=torch.float)

        if self.config.domain_rand.born_offset:
            self.log_dict["born_offset"] = torch.tensor(self.born_distance, dtype=torch.float)
        if self.config.domain_rand.born_heading:
            self.log_dict["born_heading"] = torch.tensor(self.born_heading_degree, dtype=torch.float)
        if self.config.domain_rand.push_robots:
            self.log_dict["push_robots"] = torch.tensor(self.push_robots_interval, dtype=torch.float)

        if self.config.domain_rand.motion_time_advance:
            self.log_dict["motion_time_advance"] = torch.tensor(self.advance_initial_degree, dtype=torch.float)
            self.log_dict["appaly_advance"] = torch.tensor(self.appaly_advance_interval, dtype=torch.float)
        self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
        self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()
        # self._update_contact()

    def _log_motion_tracking_info(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        joint_pos_diff = self.dif_joint_angles

        upper_body_diff_norm = upper_body_diff.norm(dim=-1).mean()
        lower_body_diff_norm = lower_body_diff.norm(dim=-1).mean()
        vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()

        self.log_dict["upper_body_diff_norm"] = upper_body_diff_norm
        self.log_dict["lower_body_diff_norm"] = lower_body_diff_norm
        self.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm
        self.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm
        
        if 'enable' in self.config.soft_dynamic_correction and self.config.soft_dynamic_correction.enable:
            self.log_dict["sdc_alpha"] = torch.tensor(self.sdc_alpha, dtype=torch.float, device=self.device)
        if 'adaptive_tracking_sigma' in self.config.rewards and self.config.rewards.adaptive_tracking_sigma.enable:
            for key, value in self._reward_error_ema.items():
                self.log_dict['error_ema_'+key] = torch.tensor(value, dtype=torch.float, device=self.device)
                self.log_dict['adp_sigma_'+key] = torch.tensor(self.config.rewards.reward_tracking_sigma[key], dtype=torch.float, device=self.device)
        

    def _draw_debug_vis(self):
        if self.config.simulator.config.name == "mjlab":
            self.simulator.clear()
            qpos = torch.cat([self.ref_root_pos, self.ref_root_rot, self.ref_joint_pos], dim=-1).squeeze(0)
            self.simulator.debug_draw(qpos)
            return
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
        # for env_id in range(self.num_envs):
            if not self.config.use_teleop_control:
                # draw marker joints
                for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
                    if self.config.robot.motion.visualization.customize_color:
                        color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                    else:
                        color_inner = (0.3, 0.3, 0.3)
                    color_inner = tuple(color_inner)

                    # import ipdb; ipdb.set_trace()
                    self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)


            else:
                # draw teleop joints
                for pos_id, pos_joint in enumerate(self.teleop_marker_coords[env_id]):
                    self.simulator.draw_sphere(pos_joint, 0.04, (0.851, 0.144, 0.07), env_id, pos_id)

    def _reset_root_states(self, env_ids):
        # reset root states according to the reference motion
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        # breakpoint()
        if self.custom_origins: # trimesh
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            offset = self.env_origins
            # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            motion_res = self.kick_motion_res()

            self.simulator.robot_root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            if self.config.domain_rand.born_offset:
                rand_num = np.random.rand()
                if rand_num < self.config.domain_rand.born_offset_possibility:
                    randomize_distance = torch_rand_float(-self.born_distance, self.born_distance, (len(env_ids), 2), device=self.device)
                    # import ipdb; ipdb.set_trace()
                    # randomize_distance = torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
                    self.simulator.robot_root_states[env_ids, :2] += randomize_distance
                    # self.root_states[env_ids, :2] += torch_rand_float(self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1], (len(env_ids), 2), device=self.device)
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_res['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'mjlab':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_res['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids]
            self.simulator.robot_root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
            if self.config.domain_rand.born_heading:
                random_angles_rad_axis = torch.zeros(len(env_ids),3, device=self.device)
                random_angles = (torch.rand((len(env_ids),), device=self.device) * (2 * self.born_heading_degree) - self.born_heading_degree)
                # random_angles = torch_rand_float(-self.cfg.domain_rand.born_heading_degree, self.cfg.domain_rand.born_heading_degree, (len(env_ids),1), device=self.device).squeeze(-1)
                # random_angles = torch.rand(len(env_ids)) * 0 - 180
                # import ipdb; ipdb.set_trace()
                random_angles_rad = torch.deg2rad(random_angles)
                # random_angles_rad = random_angles * 0
                # print("random_angles_rad_axis shape", random_angles_rad_axis.shape)
                # print("env_ids= ", env_ids)
                
                random_angles_rad_axis[:, 0] = random_angles_rad
                # random_angles_rad_axis[env_ids, 0] = 2.7
                if self.config.simulator.config.name == 'isaacgym':
                    self.simulator.robot_root_states[env_ids, 3:7] = apply_rotation_to_quat_z(self.simulator.robot_root_states[env_ids, 3:7], random_angles_rad_axis)
                elif self.config.simulator.config.name == 'isaacsim':
                    self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(apply_rotation_to_quat_z(wxyz_to_xyzw(self.simulator.robot_root_states[env_ids, 3:7]), random_angles_rad_axis))
                elif self.config.simulator.config.name == 'mjlab':
                    self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(apply_rotation_to_quat_z(wxyz_to_xyzw(self.simulator.robot_root_states[env_ids, 3:7]), random_angles_rad_axis))
        else:
            # motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
            # offset = self.env_origins
            # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)

            # if self.is_evaluating:
            #     self.simulator.robot_root_states[env_ids] = self.base_init_state
            # else:
                motion_res = self.kick_motion_res()



                root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
                root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
                root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
                root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level

                root_pos = motion_res['root_pos'][env_ids]
                root_rot = motion_res['root_rot'][env_ids]
                root_vel = motion_res['root_vel'][env_ids]
                root_ang_vel = motion_res['root_ang_vel'][env_ids]

                self.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise
                if self.config.domain_rand.born_offset:
                    rand_num = np.random.rand()
                    if rand_num < self.config.domain_rand.born_offset_possibility:
                        randomize_distance = torch_rand_float(-self.born_distance, self.born_distance, (len(env_ids), 2), device=self.device)
                        # import ipdb; ipdb.set_trace()
                        # randomize_distance = torch.clamp(randomize_distance,self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1])
                        self.simulator.robot_root_states[env_ids, :2] += randomize_distance
                        # self.root_states[env_ids, :2] += torch_rand_float(self.cfg.domain_rand.born_offset_range[0], self.cfg.domain_rand.born_offset_range[1], (len(env_ids), 2), device=self.device)
                if self.config.simulator.config.name == 'isaacgym':
                    # XYZW
                    self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                elif self.config.simulator.config.name == 'isaacsim':
                    # Isaac sim internally uses WXYZ
                    self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
                elif self.config.simulator.config.name == 'mjlab':
                    self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
                elif self.config.simulator.config.name == 'genesis':
                    self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                    # breakpoint() # Check Quaternion format
                elif self.config.simulator.config.name == 'mujoco':
                    self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
                    # breakpoint() # Check Quaternion format
                else:
                    raise NotImplementedError
                self.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
                self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise
                if self.config.domain_rand.born_heading:
                    random_angles_rad_axis = torch.zeros(len(env_ids),3, device=self.device)
                    random_angles = (torch.rand((len(env_ids),), device=self.device) * (2 * self.born_heading_degree) - self.born_heading_degree)
                    # random_angles = torch_rand_float(-self.cfg.domain_rand.born_heading_degree, self.cfg.domain_rand.born_heading_degree, (len(env_ids),1), device=self.device).squeeze(-1)
                    # random_angles = torch.rand(len(env_ids)) * 0 - 180
                    # import ipdb; ipdb.set_trace()
                    random_angles_rad = torch.deg2rad(random_angles)
                    # random_angles_rad = random_angles * 0
                    # print("random_angles_rad_axis shape", random_angles_rad_axis.shape)
                    # print("env_ids= ", env_ids)
                    
                    random_angles_rad_axis[:, 0] = random_angles_rad
                    # random_angles_rad_axis[env_ids, 0] = 2.7

                    if self.config.simulator.config.name == 'isaacgym':
                        self.simulator.robot_root_states[env_ids, 3:7] = apply_rotation_to_quat_z(self.simulator.robot_root_states[env_ids, 3:7], random_angles_rad_axis)
                    elif self.config.simulator.config.name == 'isaacsim':
                        self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(apply_rotation_to_quat_z(wxyz_to_xyzw(self.simulator.robot_root_states[env_ids, 3:7]), random_angles_rad_axis))
                    elif self.config.simulator.config.name == 'mjlab':
                        self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(apply_rotation_to_quat_z(wxyz_to_xyzw(self.simulator.robot_root_states[env_ids, 3:7]), random_angles_rad_axis))
                    elif self.config.simulator.config.name == 'mujoco':
                        self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(apply_rotation_to_quat_z(wxyz_to_xyzw(self.simulator.robot_root_states[env_ids, 3:7]), random_angles_rad_axis))
        self.simulator.robot_root_states[env_ids, 2] += 0.05 # in case under the terrain

    def small_random_quaternions(self, n, max_angle):
            # XYZW
            axis = torch.randn((n, 3), device=self.device)
            axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
            angles = max_angle * torch.rand((n, 1), device=self.device)
            
            # Convert angle-axis to quaternion
            sin_half_angle = torch.sin(angles / 2)
            cos_half_angle = torch.cos(angles / 2)
            
            q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)  
            return q

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # print("DEBUG: reset", len(self.motions_for_saving['dof']))
        # motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times # next frames so +1
        # offset = self.env_origins
        # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        # if self.is_evaluating:
        #     self.simulator.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.9, 1.1, (len(env_ids), self.num_dof), device=str(self.device))            
        #     self.simulator.dof_vel[env_ids] = 0.
        # else:
        motion_res = self.kick_motion_res()
        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        dof_pos = motion_res['dof_pos'][env_ids]
        dof_vel = motion_res['dof_vel'][env_ids]
        self.simulator.dof_pos[env_ids] = dof_pos 
        self.simulator.dof_vel[env_ids] = dof_vel 


    _kick_motion_res_counter = -1
    _kick_motion_res_buffer: Optional[Dict[str, torch.Tensor]] = None  #全局变量，仅仅是防止同一次step下的重复采样
    def kick_motion_res(self) -> Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter == self.common_step_counter:
            return self._kick_motion_res_buffer # type: ignore
        
        self._kick_motion_res_counter = self.common_step_counter
        
    
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        offset = self.env_origins
        self._kick_motion_res_buffer = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        
        return self._kick_motion_res_buffer
    
    _kick_motion_res_counter_multistep = -1
    _kick_motion_res_multistep_buffer: Optional[Dict[str, torch.Tensor]] = None
    def kick_motion_res_multiplestep(self) -> Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter_multistep == self.common_step_counter:
            return self._kick_motion_res_multistep_buffer # type: ignore
        
        self._kick_motion_res_counter_multistep = self.common_step_counter
        # motion_times = torch.cat([(self.episode_length_buf + 1)* self.dt + self.motion_start_times + (i + 1) * self.motion_dt for i in range(self.config.obs.future_ref_steps)], dim=0)
        i_values = torch.arange(0, self.config.obs.future_ref_steps, device=self.device)  # [0, 1, 2, ..., K-1]
        base_time = (self.episode_length_buf + 1) * self.dt + self.motion_start_times  # shape: [num_envs]
        motion_times = (base_time.unsqueeze(0) + i_values.unsqueeze(1) * self.dt).flatten() 
        offset = self.env_origins.repeat(self.config.obs.future_ref_steps, 1)
        motion_ids = self.motion_ids.repeat(self.config.obs.future_ref_steps)
        motion_buffer = self._motion_lib.get_motion_state_simple(motion_ids, motion_times, offset=offset)
        motion_buffer["rg_pos_t"] = motion_buffer["rg_pos_t"].reshape(self.config.obs.future_ref_steps, self.num_envs, -1, 3).transpose(0, 1).contiguous()
        self._kick_motion_res_multistep_buffer = motion_buffer
        return self._kick_motion_res_multistep_buffer
        # buffer = {}
        # for i in range(self.config.obs.future_ref_steps):
        #     motion_times = (self.episode_length_buf + 1)* self.dt + self.motion_start_times + (i + 1) * self.motion_dt
        #     offset = self.env_origins
        #     motion= self._motion_lib.get_motion_state_simple(self.motion_ids, motion_times, offset=offset)
        #     for k, v in motion.items():
        #         if k not in buffer:
        #             buffer[k] = []
        #         buffer[k].append(v.unsqueeze(1))

        # self._kick_motion_res_multistep_buffer = {k: torch.cat(v, dim=1) for k, v in buffer.items()}  # Dict[str, Tensor[B, T, D]]
        # return self._kick_motion_res_multistep_buffer
    
    _kick_motion_res_counter_advance = -1
    _kick_motion_res_buffer_advance: Optional[Dict[str, torch.Tensor]] = None  #全局变量，仅仅是防止同一次step下的重复采样
    def kick_motion_res_advance(self) -> Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter_advance == self.common_step_counter:
            return self._kick_motion_res_buffer_advance # type: ignore

        self._kick_motion_res_counter_advance = self.common_step_counter
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times + self.motion_advance_buffer * self.dt
        offset = self.env_origins
        self._kick_motion_res_buffer_advance = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        
        return self._kick_motion_res_buffer_advance
    
    _kick_motion_res_counter_multistep_advance = -1
    _kick_motion_res_multistep_buffer_advance: Optional[Dict[str, torch.Tensor]] = None
    def kick_motion_res_multiplestep_advance(self) -> Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter_multistep_advance == self.common_step_counter:
            return self._kick_motion_res_multistep_buffer_advance # type: ignore
        
        self._kick_motion_res_counter_multistep_advance = self.common_step_counter
        
        #motion_times = torch.cat([(self.episode_length_buf + 1)* self.dt + self.motion_start_times + (i + 1) * self.motion_dt + self.motion_advance_buffer * self.dt for i in range(self.config.obs.future_ref_steps)], dim=0)
        i_values = torch.arange(0, self.config.obs.future_ref_steps, device=self.device)  # [0, 1, 2, ..., K-1]
        base_time = (self.episode_length_buf + 1) * self.dt + self.motion_start_times + self.motion_advance_buffer * self.dt
        motion_times = (base_time.unsqueeze(0) + i_values.unsqueeze(1) * self.dt).flatten() 

        offset = self.env_origins.repeat(self.config.obs.future_ref_steps, 1)
        motion_ids = self.motion_ids.repeat(self.config.obs.future_ref_steps)
        motion_buffer = self._motion_lib.get_motion_state_simple(motion_ids, motion_times, offset=offset)
       
        motion_buffer["rg_pos_t"] = motion_buffer["rg_pos_t"].reshape(self.config.obs.future_ref_steps, self.num_envs, -1, 3).transpose(0, 1).contiguous()
        self._kick_motion_res_multistep_buffer_advance = motion_buffer
        return self._kick_motion_res_multistep_buffer_advance
        # buffer = {}
        # for i in range(self.config.obs.future_ref_steps):
        #     motion_times = (self.episode_length_buf + 1)* self.dt + self.motion_start_times + (i + 1) * self.motion_dt + self.motion_advance_buffer * self.dt
        #     offset = self.env_origins
        #     motion= self._motion_lib.get_motion_state_simple(self.motion_ids, motion_times, offset=offset)
        #     for k, v in motion.items():
        #         if k not in buffer:
        #             buffer[k] = []
        #         buffer[k].append(v.unsqueeze(1))

        # self._kick_motion_res_multistep_buffer_advance = {k: torch.cat(v, dim=1) for k, v in buffer.items()}  # Dict[str, Tensor[B, T, D]]
        # return self._kick_motion_res_multistep_buffer_advance
        


    # TimePortion: <12%
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        
        offset = self.env_origins
        B = self.motion_ids.shape[0]
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        
        
        # # motion_res = self._get_state_from_motionlib_cache_trimesh(self.motion_ids, motion_times, offset= offset)
        # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        motion_res = self.kick_motion_res()
        if "future_ref_steps" in self.config.obs and self.config.obs.future_ref_steps > 0:
            future_motion_res = self.kick_motion_res_multiplestep()
            # future_ref_joint_pos = future_motion_res["dof_pos"] # [num_envs, num_future_steps, num_dofs]
            # future_ref_joint_vel = future_motion_res["dof_vel"] # [num_envs, num_future_steps, num_dofs]
            future_ref_body_pos_extend = future_motion_res["rg_pos_t"] # [num_envs, num_future_steps, num_markers, 3]
            # self._obs_future_ref_dof_pos = future_ref_joint_pos.view(B, -1) # [num_envs, num_future_steps * num_dofs]
            # self._obs_future_ref_dof_vel = future_ref_joint_vel.view(B, -1) # [num_envs, num_future_steps * num_dofs]
            future_ref_body_pos_extend = future_ref_body_pos_extend.view(B, -1, 3) # [num_envs, num_future_steps * num_markers, 3]
            future_global_ref_body_pos_extend = future_ref_body_pos_extend - self.simulator.robot_root_states[:, :3].view(B, 1, 3)

            #  (Pdb) motion_res.keys()
            #  dict_keys(['root_pos', 'root_rot', 'dof_pos', 'root_vel', 'root_ang_vel', 'dof_vel', 'motion_aa', 'motion_bodies', 'rg_pos', 'rb_rot', 'body_vel', 'body_ang_vel', 'rg_pos_t', 'rg_rot_t', 'body_vel_t', 'body_ang_vel_t'])
            
                # (Pdb) print( [(k,v.shape) for k,v in motion_res.items()  ]  )
                # [('root_pos', torch.Size([1, 3])), ('root_rot', torch.Size([1, 4])), ('dof_pos', torch.Size([1, 23])), ('root_vel', torch.Size([1, 3])), ('root_ang_vel', torch.Size([1, 3])), ('dof_vel', torch.Size([1, 23])), ('motion_aa', torch.Size([1, 72])), ('motion_bodies', torch.Size([1, 17])), ('rg_pos', torch.Size([1, 24, 3])), ('rb_rot', torch.Size([1, 24, 4])), ('body_vel', torch.Size([1, 24, 3])), ('body_ang_vel', torch.Size([1, 24, 3])), ('rg_pos_t', torch.Size([1, 27, 3])), ('rg_rot_t', torch.Size([1, 27, 4])), ('body_vel_t', torch.Size([1, 27, 3])), ('body_ang_vel_t', torch.Size([1, 27, 3]))]
        motion_res_advance = self.kick_motion_res_advance()
        if "future_ref_steps" in self.config.obs and self.config.obs.future_ref_steps > 0:
            future_motion_res_advance = self.kick_motion_res_multiplestep_advance()
            # future_ref_joint_pos_advance = future_motion_res_advance["dof_pos"] # [num_envs, num_future_steps, num_dofs]
            # future_ref_joint_vel_advance = future_motion_res_advance["dof_vel"] # [num_envs, num_future_steps, num_dofs]
            future_ref_body_pos_extend_advance = future_motion_res_advance["rg_pos_t"] # [num_envs, num_future_steps, num_markers, 3]
            # self._obs_future_ref_dof_pos_advance = future_ref_joint_pos_advance.view(B, -1) # [num_envs, num_future_steps * num_dofs]
            #self._obs_future_ref_dof_vel_advance = future_ref_joint_vel_advance.view(B, -1) # [num_envs, num_future_steps * num_dofs]
            future_ref_body_pos_extend_advance = future_ref_body_pos_extend_advance.view(B, -1, 3) # [num_envs, num_future_steps * num_markers, 3]
            future_global_ref_body_pos_extend_advance = future_ref_body_pos_extend_advance - self.simulator.robot_root_states[:, :3].view(B, 1, 3)
        
        if self._motion_lib.has_contact_mask:
            self.ref_contact_mask = motion_res["contact_mask"]
            
            
        ref_body_pos_extend = motion_res["rg_pos_t"]
        self.ref_body_pos_extend[:] = ref_body_pos_extend # for visualization and analysis
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        self.ref_body_vel_extend[:] = ref_body_vel_extend
        self.ref_body_rot_extend = ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        self.ref_root_pos = motion_res["root_pos"].clone()
        self.ref_root_rot = motion_res["root_rot"][:, [3, 0, 1, 2]].clone()
        self.ref_joint_pos = ref_joint_pos.clone()
        ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]

        ref_body_pos_extend_advance = motion_res_advance["rg_pos_t"]

        ################### EXTEND Rigid body POS #####################
        rotated_pos_in_parent = my_quat_rotate( # XYZW  
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, extend_curr_pos], dim=1)

        ################### EXTEND Rigid body Rotation #####################
        extend_curr_rot = quat_mul(self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                    self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                    w_last=True).view(self.num_envs, -1, 4)
        self._rigid_body_rot_extend = torch.cat([self.simulator._rigid_body_rot, extend_curr_rot], dim=1)
        
        ################### EXTEND Rigid Body Angular Velocity #####################
        self._rigid_body_ang_vel_extend = torch.cat([self.simulator._rigid_body_ang_vel, self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]], dim=1)
    
        ################### EXTEND Rigid Body Linear Velocity #####################
        self._rigid_body_ang_vel_global = self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]
        angular_velocity_contribution = torch.cross(self._rigid_body_ang_vel_global, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
        extend_curr_vel = self.simulator._rigid_body_vel[:, self.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
        self._rigid_body_vel_extend = torch.cat([self.simulator._rigid_body_vel, extend_curr_vel], dim=1)

        ################### Compute differences #####################

        ## diff compute - kinematic position
        self.dif_global_body_pos = ref_body_pos_extend - self._rigid_body_pos_extend
        ## diff compute - kinematic rotation (use relative rotation angle instead of raw quaternion difference)
        # q_err = q_ref * conj(q_cur)
        q_err = quat_mul(
            ref_body_rot_extend,
            quat_conjugate(self._rigid_body_rot_extend),
            w_last=True,
        )
        # ensure unit quaternion to avoid numerical issues
        q_err = q_err / (q_err.norm(dim=-1, keepdim=True) + 1e-8)
        # rotation angle in [0, pi], use abs on w to handle q and -q equivalence
        cos_half_angle = torch.clamp(q_err[..., 3].abs(), -1.0 + 1e-7, 1.0 - 1e-7)
        self.dif_global_body_rot = 2.0 * torch.acos(cos_half_angle)  # [B, num_bodies + num_extend_bodies]
        ## diff compute - kinematic velocity
        self.dif_global_body_vel = ref_body_vel_extend - self._rigid_body_vel_extend
        ## diff compute - kinematic angular velocity
        
        self.dif_global_body_ang_vel = ref_body_ang_vel_extend - self._rigid_body_ang_vel_extend
        # ang_vel_reward = self._reward_teleop_body_ang_velocity_extend()



        
        ## diff compute - kinematic joint position
        self.dif_joint_angles = ref_joint_pos - self.simulator.dof_pos
        ## diff compute - kinematic joint velocity
        self.dif_joint_velocities = ref_joint_vel - self.simulator.dof_vel

        # print(self.dif_joint_angles.max(),self.dif_global_body_pos.norm(dim=-1).max())

        # breakpoint() ## Examine the data structure of ref state
        

        # marker_coords for visualization
        self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3)

        env_batch_size = self.simulator._rigid_body_pos.shape[0]
        num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]

        heading_inv_rot = calc_heading_quat_inv(self.base_quat.clone(), w_last=True)
        heading_inv_rot_expand  = heading_inv_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)
        self.heading_inv_rot = heading_inv_rot
        heading_rot = calc_heading_quat(self.base_quat.clone(), w_last=True)
        heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, num_rigid_bodies, -1).reshape(-1, 4)


        self.relyaw = self.rpy[:,2:3] - self.ref_init_rpy[0,2]
        relyaw_heading_inv_quat = calc_yaw_heading_quat_inv(self.relyaw)
        relyaw_heading_inv_quat_expand = relyaw_heading_inv_quat.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)
        # print(self.relyaw[0])

        
        
        ########## Local Diff Rigid Body Pos ##########
            # Meaning: This section computes the local difference rigid body positions relative to the robot's root frame.
            # i.e. what's the diff of each rigid body's position in the local frame?
        dif_global_body_pos_for_obs_compute = ref_body_pos_extend.view(env_batch_size, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, -1, 3)
        dif_global_body_pos_for_obs_compute_advance = ref_body_pos_extend_advance.view(env_batch_size, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, -1, 3)
        dif_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_pos_for_obs_compute.view(-1, 3))
        dif_local_body_pos_flat_advance = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), dif_global_body_pos_for_obs_compute_advance.view(-1, 3))
        self._obs_dif_local_rigid_body_pos = dif_local_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        self._obs_dif_local_rigid_body_pos_advance = dif_local_body_pos_flat_advance.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)


        if "future_ref_steps" in self.config.obs and self.config.obs.future_ref_steps > 0:
            future_dif_global_body_pos = future_ref_body_pos_extend.view(env_batch_size, self.config.obs.future_ref_steps, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, 1, -1, 3)
            future_dif_global_body_pos_advance = future_ref_body_pos_extend_advance.view(env_batch_size, self.config.obs.future_ref_steps, -1, 3) - self._rigid_body_pos_extend.view(env_batch_size, 1, -1, 3)
            heading_inv_future = heading_inv_rot.unsqueeze(1).expand(-1, self.config.obs.future_ref_steps * (self.num_extend_bodies + num_rigid_bodies), -1).reshape(-1, 4)
            future_dif_local_body_pos = my_quat_rotate(heading_inv_future.view(-1, 4), future_dif_global_body_pos.view(-1, 3))
            future_dif_local_body_pos_advance = my_quat_rotate(heading_inv_future.view(-1, 4), future_dif_global_body_pos_advance.view(-1, 3))
            self._obs_future_dif_local_rigid_body_pos = future_dif_local_body_pos.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
            self._obs_future_dif_local_rigid_body_pos_advance = future_dif_local_body_pos_advance.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

            
        ########## Local Ref Rigid Body Pos ##########
            # Meaning: This section computes the local reference rigid body positions relative to the robot's root frame.
            # It first calculates the global positions relative to the root by subtracting the root position,
            # then transforms these positions into the local frame using the inverse heading rotation.
            # The result is stored in self._obs_local_ref_rigid_body_pos for observation purposes.
        global_ref_rigid_body_pos = ref_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)  # preserves the body position
        global_ref_rigid_body_pos_advance = ref_body_pos_extend_advance.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3) # preserves the body position

        if "future_ref_steps" in self.config.obs and self.config.obs.future_ref_steps > 0:
            heading_inv_future = heading_inv_rot.unsqueeze(1).expand(-1, self.config.obs.future_ref_steps * (self.num_extend_bodies + num_rigid_bodies), -1).reshape(-1, 4)
            future_local_ref_body_pos_flat = my_quat_rotate(heading_inv_future.view(-1, 4), future_global_ref_body_pos_extend.view(-1, 3))
            future_local_ref_body_pos_flat_advance = my_quat_rotate(heading_inv_future.view(-1, 4), future_global_ref_body_pos_extend_advance.view(-1, 3))
            self._obs_future_local_ref_body_pos_extend = future_local_ref_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
            self._obs_future_local_ref_body_pos_extend_advance = future_local_ref_body_pos_flat_advance.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
        local_ref_rigid_body_pos_flat_advance = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos_advance.view(-1, 3))
        self._obs_local_ref_rigid_body_pos = local_ref_rigid_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        self._obs_local_ref_rigid_body_pos_advance = local_ref_rigid_body_pos_flat_advance.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        local_body_pos = self._rigid_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)
        local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_body_pos.view(-1, 3))
        self.local_body_pos = local_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        local_body_vel = self._rigid_body_vel_extend.view(env_batch_size, -1, 3).clone()
        local_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_body_vel.view(-1, 3))
        self.local_body_vel = local_body_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        local_body_rot = self._rigid_body_rot_extend.view(env_batch_size, -1, 4).clone()
        local_body_rot_flat = quat_mul(heading_inv_rot_expand.view(-1, 4), local_body_rot.view(-1, 4), w_last=True)
        self.local_body_rot = local_body_rot_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*4)

        local_body_rot_vel = self._rigid_body_ang_vel_extend.view(env_batch_size, -1, 3).clone()
        local_body_rot_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_body_rot_vel.view(-1, 3))
        self.local_body_rot_vel = local_body_rot_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)


        global_ref_body_vel = ref_body_vel_extend.view(env_batch_size, -1, 3)
        local_ref_rigid_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_body_vel.view(-1, 3))

        self._obs_local_ref_rigid_body_vel = local_ref_rigid_body_vel_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        self._obs_global_ref_rigid_body_vel = global_ref_body_vel.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        
        self._obs_local_ref_rigid_body_pos_relyaw = my_quat_rotate(relyaw_heading_inv_quat_expand.view(-1, 4), 
                                                                   global_ref_body_vel.view(-1, 3)).view(env_batch_size, -1)
        

        ######################VR 3 point ########################
        if not self.config.use_teleop_control:
            ref_vr_3point_pos = ref_body_pos_extend.view(env_batch_size, -1, 3)[:, self.motion_tracking_id, :]
        else:
            ref_vr_3point_pos = self.teleop_marker_coords
        vr_2root_pos = (ref_vr_3point_pos - self.simulator.robot_root_states[:, 0:3].view(env_batch_size, 1, 3))
        heading_inv_rot_vr = heading_inv_rot.repeat(3,1)
        self._obs_vr_3point_pos = my_quat_rotate(heading_inv_rot_vr.view(-1, 4), vr_2root_pos.view(-1, 3)).view(env_batch_size, -1)
        
        
        
        
        #################### Deepmimic phase ###################### 

        self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        self._ref_motion_phase = motion_times / self._ref_motion_length
        if not (torch.all(self._ref_motion_phase >= 0) and torch.all(self._ref_motion_phase <= 1.05)): # hard coded 1.05 because +1 will exceed 1
            max_phase = self._ref_motion_phase.max()
            # import ipdb; ipdb.set_trace()
        self._ref_motion_phase = self._ref_motion_phase.unsqueeze(1)
        # print(f"ref_motion_phase: {self._ref_motion_phase[0].item():.2f}")
        # print(f"ref_motion_length: {self._ref_motion_length[0].item():.2f}")
        
        self._log_motion_tracking_info()

    def _pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

        # action noise
        # self.actions += torch.randn_like(self.actions) * 0.01
        # self.actions *= 1 + torch.randn_like(self.actions) * 0.01
        
        if 'only_track_leg' in self.config.robot and self.config.robot.only_track_leg:
            self.actions = torch.cat(self.actions, self.ref_joint_pos[:, 12:], dim=-1)
        self.log_dict["action_clip_frac"] = (
                self.actions.abs() == clip_action_limit
            ).sum() / self.actions.numel()

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions_after_delay = self.action_queue[torch.arange(self.num_envs), self.action_delay_idx].clone()
        else:
            self.actions_after_delay = self.actions.clone()

    # @time_prot_fn_dec_
    # TimePortion: 32%
    def _physics_step(self):
        """
            Add Soft Dynamic Correction : alpha

            For each sim step, we modify the current state toward the target state with a ratio of alpha
            alpha is predefined in the config
            
            Im:
                1. fetch ref state of next step at the beginning 
                2. for each substep:
                    a. fetch current state
                    b. compute the difference between current state and ref state
                    c. apply soft dynamic correction
                    d. run simulate
        
        """
        
        def _physics_step_SDC(self):
            assert self.config.simulator.config.name == "isaacgym", "Soft Dynamic Correction is only supported for Isaac Gym"
            
            # if self.simulator.robot_root_states[:, 2]>0.9:
            #     breakpoint()
            
            offset = self.env_origins # [N,3]
            # motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
            # motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            motion_res = self.kick_motion_res()

            
            zero_ref_state = WholeBodyState(
                root_state=torch.cat([offset, torch.zeros((self.num_envs, 10), device=self.device)], dim=-1)+self.base_init_state,
                dof_pos=torch.tile(self.default_dof_pos, (self.num_envs, 1)),
                dof_vel=torch.zeros((self.num_envs, self.num_dofs), device=self.device)
            )
            
            ref_whole_body_state = WholeBodyState(
                root_state=torch.cat([motion_res['root_pos'], motion_res['root_rot'], motion_res['root_vel'], motion_res['root_ang_vel']], dim=-1),
                dof_pos=motion_res['dof_pos'],
                dof_vel=motion_res['dof_vel']
            )
            # ref_whole_body_state = zero_ref_state
            
            
            decimation = self.config.simulator.config.sim.control_decimation
            substep_alpha = (1-(1-self.sdc_alpha)**(1/decimation))
            
            # print(f"substep_alpha: {substep_alpha}")
            
            self.render()
            for _ in range(decimation):
                
                self.simulator.refresh_sim_tensors()
                current_state = WholeBodyState(
                    root_state=self.simulator.robot_root_states,
                    dof_pos=self.simulator.dof_pos,
                    dof_vel=self.simulator.dof_vel
                )
                
                current_state.interpolate(ref_whole_body_state, substep_alpha).apply(self.simulator)
                
                # breakpoint()
                
                self._apply_force_in_physics_step()
                self.simulator.simulate_at_each_physics_step()

            ...
            
        # root_state: [N,13], 13 = 3(pos) + 4(rot) + 3(vel) + 3(ang_vel)
        # print(f"Use Soft Dynamic Correction: {self.config.soft_dynamic_correction.enable}")
        if (not 'enable' in self.config.soft_dynamic_correction or 
            not self.config.soft_dynamic_correction.enable # clip it, for faster adaptability
            ):
            return super()._physics_step()
        else:
            if (    
                (self.config.soft_dynamic_correction.type == "prob" 
                    and np.random.rand() > np.sqrt(self.sdc_alpha) )
                or self.sdc_alpha < 0.005) :
                # print(f"SDC: use physics step, alpha: {self.sdc_alpha}, prob alpha: {np.sqrt(self.sdc_alpha)}")
                return super()._physics_step()
            # print(f"SDC: use SDC step, alpha: {self.sdc_alpha}")
            return _physics_step_SDC(self)
        
        # breakpoint()


    # @time_prot_fn_dec_
    # TimePortion: 42.5%
    def _post_physics_step(self):
        super()._post_physics_step()
        
        if self.save_motion:    
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times
            # print("DEBUG: motion_times",motion_times)
            if (len(self.motions_for_saving['dof'])) == self.config.save_total_steps+3:
                self.saved_motion_dict = {}
                for k, v in self.motions_for_saving.items():
                    self.saved_motion_dict[k] = torch.stack(v[3:]).transpose(0,1).numpy()
                    print("DEBUG: ",k,self.saved_motion_dict[k].shape, self.saved_motion_dict[k].dtype)
                    
                self.saved_motion_dict['root_trans_offset'] -= self.env_origins.cpu().numpy().reshape(-1, 1, 3)
                self.saved_motion_dict['motion_times'] = torch.stack(self.motion_times_buf[3:]).transpose(0,1).numpy()
                
                dump_data = {}
                num_motions = self.num_envs 
                num_frames = len(self.saved_motion_dict['dof'][0])
                keys_to_save = self.saved_motion_dict.keys()

                # motion_key = f"all_motion{self.num_envs}" 
                # dump_data[motion_key] = {
                #     key: self.saved_motion_dict[key] for key in keys_to_save
                # }
                # dump_data[motion_key]['fps'] = 1 / self.dt
                save_path = f"{self.save_motion_dir}_{self.num_envs }x{num_frames}-{self._motion_episode_length}.pkl"
                
                print("self._motion_episode_length=", self._motion_episode_length)
                # breakpoint()
                if self._write_to_file:
                    for i in range(num_motions):
                        motion_key = f"motion{i}" 
                        dump_data[motion_key] = {
                            key: self.saved_motion_dict[key][i] for key in keys_to_save
                        }
                        dump_data[motion_key]['fps'] = 1 / self.dt

                    joblib.dump(dump_data, save_path, compress=3)
                    
                    print(colored(f"Saved motion data to {save_path}", 'green'))
                else:
                    print(colored(f"Not saving motion data to {save_path}, because {self._write_to_file=}", 'red'))
                    
                    
                    
                    
            root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
            if self.config.simulator.config.name == "isaacgym":
                root_rot = self.simulator.robot_root_states[:, 3:7].cpu() # xyzw
            elif self.config.simulator.config.name == "isaacsim":
                root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu() # wxyz to xyzw   
            elif self.config.simulator.config.name == "mjlab":
                root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu() # wxyz to xyzw
            elif self.config.simulator.config.name == "genesis":
                root_rot = self.simulator.robot_root_states[:,  3:7].cpu() # xyzw
            else:
                raise NotImplementedError
            root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float() # sRot.from_quat: need xyzw
            dof = self.simulator.dof_pos.cpu()
            # T, num_env, J, 3
            # print(self._motion_lib.mesh_parsers.dof_axis)
            pose_aa = torch.cat([root_rot_vec[:, None, :], self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None], torch.zeros((self.num_envs, self.num_augment_joint, 3))], axis = 1)
            self.motions_for_saving['root_trans_offset'].append(root_trans)
            self.motions_for_saving['root_rot'].append(root_rot)
            self.motions_for_saving['dof'].append(dof)
            self.motions_for_saving['pose_aa'].append(pose_aa)
            self.motions_for_saving['action'].append(self.actions.cpu())
            self.motions_for_saving['actor_obs'].append(self.obs_buf_dict['actor_obs'].cpu())
            self.motions_for_saving['terminate'].append(self.reset_buf.cpu())
            
            if torch.any(self.reset_buf.cpu()):
                print("DEBUG: Reset at ",len(self.motions_for_saving['dof']))
            # else:
            #     print("DEBUG: No reset at ",len(self.motions_for_saving['dof']),self.reset_buf,motion_times)
            
            self.motions_for_saving['dof_vel'].append(self.simulator.dof_vel.cpu())
            self.motions_for_saving['root_lin_vel'].append(self.simulator.robot_root_states[:, 7:10].cpu())
            self.motions_for_saving['root_ang_vel'].append(self.simulator.robot_root_states[:, 10:13].cpu())
            self.motions_for_saving['contact_mask'].append(self.contacts_filt.cpu())
            
            self.motion_times_buf.append(motion_times.cpu())

            self.start_save = True

    # ############################################################
        
    def _get_obs_dif_local_rigid_body_pos(self):
        return self._obs_dif_local_rigid_body_pos_advance
    
    def _get_obs_local_ref_rigid_body_pos(self):
        return self._obs_local_ref_rigid_body_pos_advance

    def _get_obs_local_ref_rigid_body_pos_relyaw(self):
        # print(self._obs_local_ref_rigid_body_pos_relyaw.mean(),self._obs_local_ref_rigid_body_pos_relyaw.std())
        return self._obs_local_ref_rigid_body_pos_relyaw

    def _get_obs_ref_motion_phase(self):
        # print(self._ref_motion_phase)
        return self._ref_motion_phase
    
    def _get_obs_vr_3point_pos(self):
        return self._obs_vr_3point_pos

    def _get_obs_relyaw(self):
        # print(self.relyaw[0])
        return self.relyaw

    def _get_obs_local_body_pos(self):
        return self.local_body_pos
    
    def _get_obs_local_body_vel(self):
        return self.local_body_vel
    
    def _get_obs_local_body_rot(self):
        return self.local_body_rot
    
    def _get_obs_base_quat(self):
        return self.base_quat
    
    def _get_obs_local_body_rot_vel(self):
        return self.local_body_rot_vel

    def _get_obs_root_height(self):
        return self.simulator.robot_root_states[:, 2:3]
    
    def _get_obs_dif_joint_angles(self):
        return self.dif_joint_angles

    def _get_obs_dif_joint_velocities(self):
        return self.dif_joint_velocities
    
    def _get_obs_local_ref_rigid_body_vel(self):
        return self._obs_local_ref_rigid_body_vel
    
    def _get_obs_global_ref_rigid_body_vel(self):
        return self._obs_global_ref_rigid_body_vel

    def _get_obs_indicator_guider(self):
        return torch.ones((self.num_envs, 1), device=self.device)  # Placeholder for indicator 

    def _get_obs_indicator_learner(self):
        return torch.zeros((self.num_envs, 1), device=self.device)
    
    def _get_obs_zero_vector(self):
        return torch.zeros((self.num_envs, self.config.obs.obs_dims.zero_vector), device=self.device) # zero vector for padding
    
    def _get_obs_future_ref_dof_pos(self):
        return self._obs_future_ref_dof_pos

    def _get_obs_future_ref_dof_vel(self):
        return self._obs_future_ref_dof_vel
    
    def _get_obs_future_local_ref_body_pos_extend(self):
        if 'use_transformer' in self.config and self.config.use_transformer:
            return self._obs_future_local_ref_body_pos_extend_advance.view(self.num_envs, self.config.obs.future_ref_steps, -1)
        else:
            return self._obs_future_local_ref_body_pos_extend_advance
    
    def _get_obs_future_dif_local_rigid_body_pos(self):
        if 'use_transformer' in self.config and self.config.use_transformer:
            return self._obs_future_dif_local_rigid_body_pos_advance.view(self.num_envs, self.config.obs.future_ref_steps, -1)
        else:
            return self._obs_future_dif_local_rigid_body_pos_advance
    
    def _get_obs_history_dif_body_pos(self):
        return self._rigid_body_pos_extend.clone().reshape(self.num_envs, -1)
    
    def _get_obs_history_body_pos(self):
        return self._rigid_body_pos_extend.clone().reshape(self.num_envs, -1)

    ######################### Observations #########################
    def _get_obs_history_actor(self,):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        # history_tensors = []
        if 'use_transformer' in self.config and self.config.use_transformer:
            transformer_history=[]
        else:
            history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]

            if key == "history_dif_body_pos":
                num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]
                history_body_pos = history_tensor.view(self.num_envs, history_length, -1, 3)
                history_body_pos_diff = history_body_pos - self._rigid_body_pos_extend.view(self.num_envs, 1, -1, 3) 
                heading_inv_rot = self.heading_inv_rot
                heading_inv_history = heading_inv_rot.unsqueeze(1).expand(-1, history_length * (self.num_extend_bodies + num_rigid_bodies), -1).reshape(-1, 4)
                history_tensor = my_quat_rotate(heading_inv_history.view(-1, 4), history_body_pos_diff.view(-1, 3)).view(self.num_envs, history_length, -1)
            if key == "history_body_pos":
                num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]
                history_body_pos = history_tensor.view(self.num_envs, -1, 3)
                history_body_pos_diff = history_body_pos - self.simulator.robot_root_states[:, :3].view(self.num_envs, 1, 3)
                heading_inv_rot = self.heading_inv_rot
                heading_inv_history = heading_inv_rot.unsqueeze(1).expand(-1, history_length * (self.num_extend_bodies + num_rigid_bodies), -1).reshape(-1, 4)
                history_tensor = my_quat_rotate(heading_inv_history.view(-1, 4), history_body_pos_diff.view(-1, 3)).view(self.num_envs, history_length, -1)
                
            if 'use_transformer' in self.config and self.config.use_transformer:
                history_tensor_transformer = history_tensor.clone()
                transformer_history.append(history_tensor_transformer)
            else:
                history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
                history_tensors.append(history_tensor)
        if 'use_transformer' in self.config and self.config.use_transformer:
            return torch.cat(transformer_history, dim=2)
        else:
            return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self,):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        if 'use_transformer' in self.config and self.config.use_transformer:
            transformer_history=[]
        else:
            history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]

            if key == "history_dif_body_pos":
                num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]
                history_body_pos = history_tensor.view(self.num_envs, history_length, -1, 3)
                history_body_pos_diff = history_body_pos - self._rigid_body_pos_extend.view(self.num_envs, 1, -1, 3) 
                heading_inv_rot = self.heading_inv_rot
                heading_inv_history = heading_inv_rot.unsqueeze(1).expand(-1, history_length * (self.num_extend_bodies + num_rigid_bodies), -1).reshape(-1, 4)
                history_tensor = my_quat_rotate(heading_inv_history.view(-1, 4), history_body_pos_diff.view(-1, 3)).view(self.num_envs, history_length, -1)
            if key == "history_body_pos":
                num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]
                history_body_pos = history_tensor.view(self.num_envs, -1, 3)
                history_body_pos_diff = history_body_pos - self.simulator.robot_root_states[:, :3].view(self.num_envs, 1, 3)
                heading_inv_rot = self.heading_inv_rot
                heading_inv_history = heading_inv_rot.unsqueeze(1).expand(-1, history_length * (self.num_extend_bodies + num_rigid_bodies), -1).reshape(-1, 4)
                history_tensor = my_quat_rotate(heading_inv_history.view(-1, 4), history_body_pos_diff.view(-1, 3)).view(self.num_envs, history_length, -1)
                
            if 'use_transformer' in self.config and self.config.use_transformer:
                history_tensor_transformer = history_tensor.clone()
                transformer_history.append(history_tensor_transformer)
            else:
                history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
                history_tensors.append(history_tensor)
        if 'use_transformer' in self.config and self.config.use_transformer:
            return torch.cat(transformer_history, dim=2)
        else:
            return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_amp(self,):
        assert "history_amp" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_amp']
        history_key_list = history_config.keys()
        # history_tensors = []
        history_tensors = []
        for key in history_config.keys():
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    ###############################################################
    
    def _init_adaptive_sigma(self):
        if not 'adaptive_tracking_sigma' in self.config.rewards or not self.config.rewards.adaptive_tracking_sigma.enable:
            self._update_adaptive_sigma = lambda *args, **kwargs: None
            logger.info("Adaptive tracking sigma Disabled")
            return
        logger.info("Adaptive tracking sigma Enabled")
        self._reward_error_ema = dict()
        for key, value in self.config.rewards.reward_tracking_sigma.items():
            self._reward_error_ema[key] = value
        
        ...
    
    def _update_adaptive_sigma(self, error:torch.Tensor, term:str):
        alpha = self.config.rewards.adaptive_tracking_sigma.alpha
        scale = (self.config.rewards.adaptive_tracking_sigma.scale 
                    if 'scale' in self.config.rewards.adaptive_tracking_sigma else 1.0)
        adptype = (self.config.rewards.adaptive_tracking_sigma.type 
                    if 'type' in self.config.rewards.adaptive_tracking_sigma else "origin")
        
        self._reward_error_ema[term] = self._reward_error_ema[term] * (1-alpha) + error.mean().item() * alpha
        
        if adptype == "scale":
            self.config.rewards.reward_tracking_sigma[term] = min(self._reward_error_ema[term] * scale, 
                                                                self.config.rewards.reward_tracking_sigma[term])
        elif adptype == "mean":
            self.config.rewards.reward_tracking_sigma[term] = (min(self._reward_error_ema[term], 
                                                                self.config.rewards.reward_tracking_sigma[term]) +
                                                                self._reward_error_ema[term]) / 2
        elif adptype == "origin":
            self.config.rewards.reward_tracking_sigma[term] = min(self._reward_error_ema[term], 
                                                                self.config.rewards.reward_tracking_sigma[term])
        
        # 确保 sigma 不小于配置的最小值
        if ('reward_tracking_min_sigma' in self.config.rewards and 
            term in self.config.rewards.reward_tracking_min_sigma):
            min_sigma = self.config.rewards.reward_tracking_min_sigma[term]
            self.config.rewards.reward_tracking_sigma[term] = max(
                self.config.rewards.reward_tracking_sigma[term], 
                min_sigma
            )
        if ('reward_tracking_max_sigma' in self.config.rewards and 
            term in self.config.rewards.reward_tracking_max_sigma):
            max_sigma = self.config.rewards.reward_tracking_max_sigma[term]
            self.config.rewards.reward_tracking_sigma[term] = min(
                self.config.rewards.reward_tracking_sigma[term], 
                max_sigma
            )
        ...

    def _reward_teleop_contact_mask(self):
        cur_contact_mask = self.contacts_filt
        ref_contact_mask = self.ref_contact_mask
        
        error_contact_mask = (cur_contact_mask - ref_contact_mask).abs()

        rew = 1 - error_contact_mask.mean(dim=-1)
        # print(f"{rew.mean()=} | {rew.std()=} | {error_contact_mask.mean()=} | {error_contact_mask.std()=}")
        return rew
    
    
    def _reward_teleop_contact_mask_v2(self):
        cur_contact_mask = self.contacts_filt
        ref_contact_mask = self.ref_contact_mask
        
        error_contact_mask = (cur_contact_mask - ref_contact_mask).abs()

        rew = 0.5 - error_contact_mask.mean(dim=-1)
        # print(f"{rew.mean()=} | {rew.std()=} | {error_contact_mask.mean()=} | {error_contact_mask.std()=}")
        return rew
    
    
    
    def _reward_teleop_body_position_extend(self):
        # self._print_tracking()
        
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)
            
        
        # print(f"{diff_body_pos_dist_upper=}")
        
        # print(f"{lower_body_diff.norm(dim=-1).norm(dim=-1).item()=:.6f}") 
        
        # max_body_pos = self.dif_global_body_pos.norm(dim=-1).max(dim=-1)[0]
        # print(f"{max_body_pos=}")

        
        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight
    
        self._update_adaptive_sigma(diff_body_pos_dist_upper, 'teleop_upper_body_pos')
        self._update_adaptive_sigma(diff_body_pos_dist_lower, 'teleop_lower_body_pos')

        return r_body_pos
    
    def _reward_teleop_vr_3point(self):
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        vr_3point_dist = (vr_3point_diff**2).mean(dim=-1).mean(dim=-1)
        # print(f"{vr_3point_dist=}")
        r_vr_3point = torch.exp(-vr_3point_dist / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)
        
        self._update_adaptive_sigma(vr_3point_dist, 'teleop_vr_3point_pos')
        return r_vr_3point

    def _reward_teleop_body_position_feet(self):

        feet_diff = self.dif_global_body_pos[:, self.feet_body_indices, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        # feet_dist_feet = (feet_diff**2).mean(dim=-1)
        # print(f"{feet_dist=} | {feet_dist_feet.max()=} | {feet_dist_feet.min()=}")
        r_feet = torch.exp(-feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos)
        
        self._update_adaptive_sigma(feet_dist, 'teleop_feet_pos')
        return r_feet
    
    def _reward_teleop_body_rotation_extend(self):
        # self.dif_global_body_rot is the per-body rotation angle difference in radians
        rotation_diff = self.dif_global_body_rot  # [B, num_bodies + num_extend_bodies]
        diff_body_rot_dist = (rotation_diff**2).mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.config.rewards.reward_tracking_sigma.teleop_body_rot)
        
        self._update_adaptive_sigma(diff_body_rot_dist, 'teleop_body_rot')
        return r_body_rot
    
    def _reward_teleop_body_rotation_feet(self):

        feet_rot = self.dif_global_body_rot[:, self.feet_indices]  # [B, num_feet]
        feet_rot_dist = (feet_rot**2).mean(dim=-1)
        r_feet_rot = torch.exp(-feet_rot_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_rot)
        
        self._update_adaptive_sigma(feet_rot_dist, 'teleop_feet_rot')
        return r_feet_rot

    def _reward_teleop_body_velocity_extend(self):
        velocity_diff = self.dif_global_body_vel    
        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        # print(f"{diff_body_vel_dist=}")
        r_body_vel = torch.exp(-diff_body_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_vel)
        
        self._update_adaptive_sigma(diff_body_vel_dist, 'teleop_body_vel')
        return r_body_vel
    
    def _reward_teleop_radial_body_velocity_extend(self):
        velocity_diff = self.dif_global_body_vel.reshape(self.num_envs, -1)
        cur_velocity = self._rigid_body_vel_extend.reshape(self.num_envs, -1)
        ref_velocity = velocity_diff + cur_velocity
        
        potential_velocity = radial_velocity_potential(cur_velocity, ref_velocity)
        return potential_velocity
    
    def _reward_teleop_body_ang_velocity_extend(self):
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        # print(f"{diff_body_ang_vel_dist=}")
        r_body_ang_vel = torch.exp(-diff_body_ang_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel)
        
        self._update_adaptive_sigma(diff_body_ang_vel_dist, 'teleop_body_ang_vel')
        return r_body_ang_vel

    def _reward_teleop_max_joint_position(self):
        joint_pos_diff:torch.Tensor = self.dif_joint_angles
        max_diff_joint_pos:torch.Tensor = (joint_pos_diff.abs()).max(dim=-1)[0]
        r_max_joint_pos = torch.exp(-max_diff_joint_pos / self.config.rewards.reward_tracking_sigma.teleop_max_joint_pos)
        
        self._update_adaptive_sigma(max_diff_joint_pos, 'teleop_max_joint_pos')
        # print(f"{max_diff_joint_pos=} \t| {r_max_joint_pos=}")
            # max diff joint pos: 0.2->0.5
        return r_max_joint_pos

    def _reward_teleop_joint_position(self):
        joint_pos_diff = self.dif_joint_angles
        diff_joint_pos_dist = (joint_pos_diff**2).mean(dim=-1)
        
        # diff_joint = (joint_pos_diff**2)
        # print(f"{diff_joint_pos_dist=} \t| {diff_joint.max()=} \t| {diff_joint.min()=}")
        r_joint_pos = torch.exp(-diff_joint_pos_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_pos)
        
        self._update_adaptive_sigma(diff_joint_pos_dist, 'teleop_joint_pos')
        # self._reward_feet_air_time() #DEBUG:
        return r_joint_pos
    
    def _reward_teleop_joint_velocity(self):
        joint_vel_diff = self.dif_joint_velocities
        diff_joint_vel_dist = (joint_vel_diff**2).mean(dim=-1)
        r_joint_vel = torch.exp(-diff_joint_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)
        
        self._update_adaptive_sigma(diff_joint_vel_dist, 'teleop_joint_vel')
        # print(f"{diff_joint_vel_dist.item()=} \t | {r_joint_vel=} \t | {self.dif_joint_velocities.abs().max(dim=-1)[0]}")
        return r_joint_vel
    
    def _reward_teleop_radial_joint_velocity(self):
        velocity_diff = self.dif_joint_velocities
        cur_velocity = self.simulator.dof_vel
        ref_velocity = velocity_diff + cur_velocity
        
        potential_velocity = radial_velocity_potential(cur_velocity, ref_velocity)
        return potential_velocity
    
    
    def setup_visualize_entities(self):
        if self.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass
    


    ## exbody2 rewards, found in envs.locomotion
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        # print("contact", contact, end='\r', flush=True)
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        # self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(torch.clamp(10 * (self.feet_air_time - self.config.rewards.desired_feet_air_time_min) * first_contact, max=0.003) + \
                                torch.clamp(10 * (- self.feet_air_time + self.config.rewards.desired_feet_air_time_max) * first_contact, max=0.003), dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.ref_body_vel_extend[:, 0, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        
        return rew_airTime
    
    def _reward_teleop_feet_height(self):
        feet_diff = self.dif_global_body_pos[:, self.feet_body_indices, 2]
        feet_height_dist = (feet_diff**2).mean(dim=-1)
        # r_feet = torch.exp(-feet_height_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_height)
        # self._update_adaptive_sigma(feet_height_dist, 'teleop_feet_height')
        r_feet = 10 * feet_height_dist - 0.009
        return r_feet
    
    # def _reward_feet_min_height_for_this_air(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     from_air_to_contact = torch.logical_and(contact_filt, ~self.last_contacts_filt)

    #     self.feet_air_min_height = torch.max(self.feet_air_min_height, self.simulator._rigid_body_pos[:, self.feet_indices, 2])
        
    #     rew_feet_min_height = torch.sum((10 * torch.clamp(self.feet_air_min_height - 0.17, max=0.05)) * from_air_to_contact, dim=1) # reward only on first contact with the ground
    #     self.feet_air_min_height *= ~contact_filt
    #     return rew_feet_min_height
    
    def _reward_penalty_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.locomotion_max_contact_force).clip(min=0.), dim=1)
    
    def _reward_penalty_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1)