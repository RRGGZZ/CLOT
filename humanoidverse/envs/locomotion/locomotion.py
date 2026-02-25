
from humanoidverse.utils.lpf import ActionFilterButterTorch

from time import time
from warnings import WarningMessage
import numpy as np
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
from rich.progress import Progress

from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase



class Locomotion(LeggedRobotBase):

    def __init__(self, config, device):
        super().__init__(config, device)
        if "action_filt" in self.config.robot.control and self.config.robot.control.action_filt:
            self.action_filter = ActionFilterButterTorch(lowcut=np.zeros(self.num_envs*self.dim_actions),
                                                        highcut=np.ones(self.num_envs*self.dim_actions) * self.config.robot.control.action_cutfreq, 
                                                        sampling_rate=1./self.dt, num_joints=self.num_envs * self.dim_actions, 
                                                        device=self.device)
        self.upper_left_arm_dof_names = self.config.robot.upper_left_arm_dof_names
        self.upper_right_arm_dof_names = self.config.robot.upper_right_arm_dof_names
        self.upper_left_arm_joint_indices = [self.dof_names.index(dof) for dof in self.upper_left_arm_dof_names]
        self.upper_right_arm_joint_indices = [self.dof_names.index(dof) for dof in self.upper_right_arm_dof_names]
        self.arm_joint_indices = self.upper_left_arm_joint_indices + self.upper_right_arm_joint_indices

        self.lower_left_leg_dof_names = self.config.robot.lower_left_leg_dof_names
        self.lower_right_leg_dof_names = self.config.robot.lower_right_leg_dof_names
        self.lower_left_leg_joint_indices = [self.dof_names.index(dof) for dof in self.lower_left_leg_dof_names]
        self.lower_right_leg_joint_indices = [self.dof_names.index(dof) for dof in self.lower_right_leg_dof_names]
        self.leg_joint_indices = self.lower_left_leg_joint_indices + self.lower_right_leg_joint_indices

        self.waist_dof_names = self.config.robot.waist_dof_names
        self.waist_joint_indices = [self.dof_names.index(dof) for dof in self.waist_dof_names]

        self.ankle_dof_names = self.config.robot.ankle_dof_names
        self.ankle_joint_indices = [self.dof_names.index(dof) for dof in self.ankle_dof_names]

        self.hip_dof_names = self.config.robot.hip_dof_names
        self.hip_joint_indices = [self.dof_names.index(dof) for dof in self.hip_dof_names]

        self.set_is_evaluating()
    
    def _get_phase(self):
        cycle_time = self.config.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase
    
    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask
    
    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.simulator.dof_pos)
        scale_1 = self.config.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        self.ref_dof_pos[:, 15] = -sin_pos * scale_1
        self.ref_dof_pos[:, 19] = sin_pos * scale_1

        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 0] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1

        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 6] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = -sin_pos_r * scale_1

        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0


    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges
        self.first_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.avg_yaw_vel = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        self.use_command_curriculum = self.config.rewards.command_curriculum
        if self.use_command_curriculum:
            self.command_scale = self.config.rewards.command_initial_value
        else:
            self.command_scale = 1.0

    
    def _setup_simulator_control(self):
        self.simulator.commands = self.commands
    
    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        super()._update_tasks_callback()

        # commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading), 
            self.command_ranges["ang_vel_yaw"][0], 
            self.command_ranges["ang_vel_yaw"][1]
        )
    
    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0]*self.command_scale, self.command_ranges["lin_vel_x"][1]*self.command_scale, (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0]*self.command_scale, self.command_ranges["lin_vel_y"][1]*self.command_scale, (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0]*self.command_scale, self.command_ranges["heading"][1]*self.command_scale, (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
    
    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        if self.use_command_curriculum:
            self._update_command_curriculum()
        self._resample_commands(env_ids)
    
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.commands = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands
    
    def next_task(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_commands(env_ids)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading), 
            self.command_ranges["ang_vel_yaw"][0], 
            self.command_ranges["ang_vel_yaw"][1]
        )
        print(f"DEBUG: commands = {self.commands}")
    
    def _update_command_curriculum(self):
        if self.average_episode_length < self.config.rewards.command_curriculum_level_down_threshold:
            self.command_scale *= (1 - self.config.rewards.command_curriculum_degree)
        elif self.average_episode_length > self.config.rewards.command_curriculum_level_up_threshold:
            self.command_scale *= (1 + self.config.rewards.command_curriculum_degree)
        self.command_scale = np.clip(self.command_scale, self.config.rewards.command_min_value, self.config.rewards.command_max_value).item()
    
    def _compute_reward(self):
        super()._compute_reward()
        if self.use_command_curriculum:
            self.log_dict["command_scale"] = torch.tensor(self.command_scale, dtype=torch.float)
    
    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf)
        self.avg_yaw_vel[env_ids] = 0.0



    ########################### TRACKING REWARDS ###########################

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.config.rewards.reward_tracking_sigma)


    def _reward_contact_momentum(self):
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        feet_contact_momentum_z = torch.abs(foot_vel[:, :, 2] * self.simulator.contact_forces[:, self.feet_indices, 2])
        return torch.sum(feet_contact_momentum_z, dim=1)


    
    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.config.rewards.reward_tracking_sigma)

    ########################### PENALTY REWARDS ###########################

    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_penalty_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_penalty_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.simulator.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(torch.sum(torch.square(self.commands[:,0:2]), dim=-1), min=0.01)        
    
    def _reward_penalty_ang_vel_xy_torso(self):
        torso_ang_vel = quat_rotate_inverse(self.simulator._rigid_body_rot[:, self.torso_index], self.simulator._rigid_body_ang_vel[:, self.torso_index])
        return torch.sum(torch.square(torso_ang_vel[:, :2]), dim=1)

    # def _reward_base_height(self):
    #     base_height = self.simulator.robot_root_states[:, 2]
    #     return torch.abs(base_height - self.config.rewards.base_height_target)

    def _reward_feet_clearance(self):
        foot_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices]

        cur_footpos_translated = foot_pos - self.simulator.robot_root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = foot_vel - self.simulator.robot_root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.config.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_smoothness(self):
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques / self.p_gains.unsqueeze(0)), dim=1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        # self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(torch.clamp(10 * (self.feet_air_time - self.config.rewards.desired_feet_air_time_min) * first_contact, max=0.003) + \
                                torch.clamp(10 * (- self.feet_air_time + self.config.rewards.desired_feet_air_time_max) * first_contact, max=0.003), dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.simulator.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1) -  self.config.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)


    def _reward_no_fly(self):
        contacts = self.simulator.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(rew_no_fly, 1. * (torch.norm(self.commands[:, :2], dim=1) < 0.1)) # full reward for zero command
        return rew_no_fly
    
    def _reward_joint_tracking_error(self):
        return torch.sum(torch.square(self.joint_pos_target - self.simulator.dof_pos), dim=-1)
    
    def _reward_joint_deviation(self):
        return torch.sum(torch.square(self.simulator.dof_pos - self.default_dof_pos), dim=-1)
    
    def _reward_arm_joint_deviation(self):
        return torch.sum(torch.square(self.simulator.dof_pos - self.default_dof_pos)[self.arm_joint_indices], dim=-1)
    
    def _reward_leg_joint_deviation(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[self.leg_joint_indices], dim=-1)
    
    def _reward_feet_distance_lateral(self):
        foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices]
        cur_footpos_translated = foot_pos - self.simulator.robot_root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        # return torch.clip(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0)
        return torch.clip(foot_leteral_dis - self.config.rewards.least_feet_distance_lateral, max=0) + torch.clip(self.config.rewards.max_feet_distance_lateral - foot_leteral_dis, max=0)
    
    def _reward_knee_distance_lateral(self):
        cur_knee_pos_translated = self.simulator._rigid_body_pos[:, self.knee_indices, :3].clone() - self.simulator.robot_root_states[:, 0:3].unsqueeze(1)
        knee_pos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            knee_pos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_translated[:, i, :])
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - knee_pos_in_body_frame[:, 1, 1])
        return torch.clamp(knee_lateral_dis - self.config.rewards.least_knee_distance_lateral, max=0)
    
    def _reward_feet_distance_lateral(self):
        foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices]
        cur_footpos_translated = foot_pos - self.simulator.robot_root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        return torch.clamp(foot_leteral_dis - self.config.rewards.least_feet_distance_lateral, max=0)
    
    def _reward_feet_slip(self): 
        # Penalize feet slipping
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(torch.norm(self.simulator._rigid_body_vel[:, self.feet_indices, :2], dim=2) * contact, dim=1)
    
    def _reward_contact_momentum(self):
        # encourage soft contacts
        feet_contact_momentum_z = torch.abs(self.simulator._rigid_body_vel[:, self.feet_indices, 2] * self.simulator.contact_forces[:, self.feet_indices, 2])
        return torch.sum(feet_contact_momentum_z, dim=1)
    
    def _reward_deviation_all_joint(self):
        return torch.sum(torch.square(self.simulator.dof_pos - self.default_dof_pos), dim=-1)
    
    def _reward_deviation_arm_joint(self):
        return torch.sum(torch.square(self.simulator.dof_pos - self.default_dof_pos)[:, self.arm_joint_indices], dim=-1)
    
    def _reward_deviation_leg_joint(self):
        return torch.sum(torch.square(self.simulator.dof_pos - self.default_dof_pos)[:, self.leg_joint_indices], dim=-1)
    
    def _reward_deviation_hip_joint(self):
        return torch.sum(torch.square(self.simulator.dof_pos - self.default_dof_pos)[:, self.hip_joint_indices], dim=-1)
    
    def _reward_deviation_waist_joint(self):
        return torch.sum(torch.square(self.simulator.dof_pos - self.default_dof_pos)[:, self.waist_joint_indices], dim=-1)
    
    def _reward_deviation_ankle_joint(self):
        return torch.sum(torch.square(self.simulator.dof_pos - self.default_dof_pos)[:, self.ankle_joint_indices], dim=-1)

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(
            self.simulator._rigid_body_vel[:, self.feet_indices, :2], dim=2
        )
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)
    

    def _get_obs_history_actor(self,):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self,):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_commands(self,):
        return self.commands[:, :3]

    
    def _reward_torso_orientation(self):
        torso_quat = self.simulator._rigid_body_rot[:, self.torso_index]
        torso_gravity = quat_rotate_inverse(torso_quat, self.gravity_vec)
        orientation = torch.exp(
            -torch.norm(torso_gravity[:, :2], dim=1) * 20
        )
        return orientation

    def _get_obs_stance_mask(self):
        stance_mask = self._get_gait_phase()
        return stance_mask.float()  # 转换为浮点数
    
    def _get_obs_contact_mask(self):
        contact_mask = self.simulator.contact_forces[:, self.feet_indices, 2] > 5.0
        return contact_mask.float()  # 转换为浮点数
    
    def _get_obs_gait_phase(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)
        gait_phase = torch.cat([sin_pos, cos_pos], dim=1)
        return gait_phase
    
    def _get_obs_dof_diff(self):
        self.compute_ref_state()
        return self.simulator.dof_pos - self.ref_dof_pos
    
    def _get_obs_avg_yaw_vel(self):
        """
        计算平均yaw角速度，使用指数移动平均进行平滑处理
        Returns: [batch_size, 1] 张量
        """
        prev_avg_yaw_vel = self.avg_yaw_vel.clone()
        self.avg_yaw_vel = (
            self.dt / self.config.rewards.cycle_time * self.base_ang_vel[:, 2]
            + (1 - self.dt / self.config.rewards.cycle_time) * prev_avg_yaw_vel
        )
        return self.avg_yaw_vel.clone().unsqueeze(-1)
    
    def _reward_joint_pos(self):

        joint_pos = self.simulator.dof_pos.clone()
        self.compute_ref_state()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target - self.default_dof_pos
        diff[:, [0, 3, 4, 6, 9, 10]] = 2 * diff[:, [0, 3, 4, 6, 9, 10]]
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 5.
        diff[:, [1, 2]] = torch.where(contact[:, 0].unsqueeze(1) == 1, diff[:, [1, 2]], 0. * diff[:, [1, 2]])
        diff[:, [7, 8]] = torch.where(contact[:, 1].unsqueeze(1) == 1, diff[:, [7, 8]], 0. * diff[:, [7, 8]])
        r = torch.exp(-1 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r
    
    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.simulator._rigid_body_pos[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.config.rewards.min_dist
        max_df = self.config.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0.0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2
    
    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.simulator._rigid_body_pos[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.config.rewards.min_dist
        max_df = self.config.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2
    
    def _reward_gait_feet_distance(self):
        leftfoot = (
            self.simulator._rigid_body_pos[:, self.feet_indices][:, 0, 0:3] - self.simulator.robot_root_states[:, 0:3]
        )
        rightfoot = (
            self.simulator._rigid_body_pos[:, self.feet_indices][:, 1, 0:3] - self.simulator.robot_root_states[:, 0:3]
        )
        leftfoot = quat_apply(quat_conjugate(self.simulator.robot_root_states[:, 3:7]), leftfoot)
        rightfoot = quat_apply(quat_conjugate(self.simulator.robot_root_states[:, 3:7]), rightfoot)
        feet_distance_y = torch.abs(leftfoot[:, 1] - rightfoot[:, 1] - 0.24)
        feet_distance = torch.abs(leftfoot[:, 1] + rightfoot[:, 1])
        return 0.5 * torch.exp(-20 * feet_distance_y)  + 0.5 * torch.exp(-20 * feet_distance)
    

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)
    
    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum(
            (
                torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1)
                - self.config.rewards.max_contact_force
            ).clip(0, 400),
            dim=1,
        )
    
    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.simulator.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, 1:3]
        right_yaw_roll = joint_diff[:, 7:9]
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 5.
        yaw_roll = torch.cat([torch.sum(torch.abs(left_yaw_roll), dim=1).unsqueeze(1), torch.sum(torch.abs(right_yaw_roll), dim=1).unsqueeze(1)], dim=1)
        yaw_roll = torch.where(contact == 1, yaw_roll, 0. * yaw_roll)
        yaw_roll = torch.sum(yaw_roll, dim=1)
        return torch.exp(-yaw_roll * 100) - 0.0 * torch.norm(joint_diff, dim=1)
    
    def _reward_default_torso(self):
        joint_diff = self.simulator.dof_pos - self.default_dof_pos
        torso_diff = -torch.norm(joint_diff[:, 12:15], dim=1)
        Pitch_diff = -torch.abs(joint_diff[:, 13])
        return torso_diff + Pitch_diff
    
    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.simulator._rigid_body_pos[:, self.feet_indices, 2] * stance_mask, dim=1
        ) / torch.sum(stance_mask, dim=1)
        base_height = self.simulator.robot_root_states[:, 2] - (measured_heights - 0.065)
        return torch.exp(
            -torch.abs(base_height - self.config.rewards.base_height_target) * 100
        )
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 5.0

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.simulator._rigid_body_pos[:, self.feet_indices, 2] - 0.065
        left_foot_height = (
            torch.mean(
                self.simulator._rigid_body_pos[:, self.feet_indices][:, 0, 2].unsqueeze(1), dim=1, )
            - 0.065
        )
        right_foot_height = (
            torch.mean(
                self.simulator._rigid_body_pos[:, self.feet_indices][:, 1, 2].unsqueeze(1),
                dim=1, )
            - 0.065
        )
        # self.feet_height += delta_z
        self.feet_height = torch.cat(
            (left_foot_height.unsqueeze(1), right_foot_height.unsqueeze(1)), dim = 1
        )
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = (
            torch.abs(self.feet_height - self.config.rewards.target_feet_height) < 0.01
        )
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos
    
    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)

        c_update = (lin_mismatch + ang_mismatch) / 2.0

        return c_update
    
    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(
            self.commands[:, 0]
        )

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1
        )
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error
    
    def _reward_feet_orien_forward(self):

        left_feet_forward = quat_rotate(
            self.simulator._rigid_body_rot[:, self.feet_indices][:, 0, :], self.forward_vec
        )
        right_feet_forward = quat_rotate(
            self.simulator._rigid_body_rot[:, self.feet_indices][:, 1, :], self.forward_vec
        )
        left_feet_forward_local = quat_apply(
            quat_conjugate(self.simulator.robot_root_states[:, 3:7]), left_feet_forward
        )
        right_feet_forward_local = quat_apply(
            quat_conjugate(self.simulator.robot_root_states[:, 3:7]), right_feet_forward
        )
        left_error_swing_rp = torch.sum(
            torch.square(self.forward_vec[:, :2] - left_feet_forward_local[:, :2]),
            dim=-1,
        )
        left_error_swing_rpy = torch.sum(
            torch.square(self.forward_vec - left_feet_forward_local), dim=-1
        )

        right_error_swing_rp = torch.sum(
            torch.square(self.forward_vec[:, :2] - right_feet_forward_local[:, :2]),
            dim=-1,
        )
        right_error_swing_rpy = torch.sum(
            torch.square(self.forward_vec - right_feet_forward_local), dim=-1
        )

        angle_mask = torch.abs(self.commands[:, 2]) > 0.1
        left_error_swing = torch.where(
            angle_mask, left_error_swing_rp, left_error_swing_rpy
        )
        right_error_swing = torch.where(
            angle_mask, right_error_swing_rp, right_error_swing_rpy
        )

        left_fori_score = torch.exp(-10.0 * left_error_swing)
        right_fori_score = torch.exp(-10.0 * right_error_swing)
        return left_fori_score + right_fori_score
    

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.simulator.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:, 1:3]
        right_yaw_roll = joint_diff[:, 7:9]
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 5.
        yaw_roll = torch.cat([torch.sum(torch.abs(left_yaw_roll), dim=1).unsqueeze(1), torch.sum(torch.abs(right_yaw_roll), dim=1).unsqueeze(1)], dim=1)
        yaw_roll = torch.where(contact == 1, yaw_roll, 0. * yaw_roll)
        yaw_roll = torch.sum(yaw_roll, dim=1)
        return torch.exp(-yaw_roll * 100) - 0.0 * torch.norm(joint_diff, dim=1)
    
    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.simulator.robot_root_states[:, 7:13] - self.simulator.robot_root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)

        term_2 = torch.sum(
            torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
            dim=1,
        )
        
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(
            torch.square((self.last_dof_vel - self.simulator.dof_vel) / self.dt), dim=1
        )
    
    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(
            -torch.sum(torch.abs(self.rpy[:, :2]), dim=1) * 10
        )
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.0
    
    def _reward_default_arm(self):
        """
        Calculates the reward for keeping arm joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.simulator.dof_pos - self.default_dof_pos
        left_yaw_roll =  joint_diff[:,20:22]
        right_yaw_roll = joint_diff[:,16:18]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100)
    
    

    

    



        
