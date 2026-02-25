import torch
import torch.optim as optim
from torch import Tensor
from collections import deque
import os
import statistics

from humanoidverse.agents.modules.ppo_modules import *
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask

import time
from lightning.fabric import Fabric
from hydra.utils import instantiate
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()

from humanoidverse.agents.mh_ppo.mh_ppo import MHPPO
from humanoidverse.utils.replay_buffer import ReplayBuffer
from humanoidverse.agents.modules.amp_discriminator import Discriminator
from humanoidverse.agents.modules.data_utils import swap_and_flatten01, compute_humanoid_observations_max

class AMP(MHPPO):
    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(self, 
                 fabric: Fabric, 
                 env: BaseTask, 
                 config,
                 log_dir=None):
        super().__init__(fabric, env, config, log_dir)
        self.discriminator_learning_rate = self.config.discriminator_learning_rate
        self.amp_replay_buffer = ReplayBuffer(self.config.discriminator_replay_size).to(
            self.device
        )
        self.extrarewbuffer = deque(maxlen=100)
        self.cur_extra_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    def setup(self):
        super().setup()
        self.discriminator: Discriminator = instantiate(
            self.config.discriminator,
            obs_dim_dict=self.algo_obs_dim_dict,
        )
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.discriminator_learning_rate)

        self.discriminator, self.discriminator_optimizer = self.fabric.setup(self.discriminator, self.discriminator_optimizer)
        self.discriminator.mark_forward_method('compute_logits')
        self.discriminator.mark_forward_method('compute_reward')

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env, self.device)
        amp_num_steps = int((self.config.discriminator_batch_size * self.config.discriminator_mini_batches - 1) // self.env.num_envs + 1)
        self.storage_amp = RolloutStorage(self.env.num_envs, amp_num_steps, self.device)
        ## Register obs keys
        if self.config.module_dict.actor.type == 'Transformer' and self.config.module_dict.critic.type == 'Transformer':
            for obs_key, obs_dim in self.algo_obs_dim_dict.items():
                input_dim_past = 0
                input_dim_future = 0
                input_dim_current = 0
                if 'amp' in obs_key:
                    self.storage_amp.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
                    # self.storage.register_key('next_'+obs_key, shape=(obs_dim,), dtype=torch.float)
                else:
                    for key in self.obs_slices[obs_key].keys():
                        if 'history' in key:
                            input_dim_past += self.obs_slices[obs_key][key][1] - self.obs_slices[obs_key][key][0]
                        elif 'future' in key:
                            input_dim_future += self.obs_slices[obs_key][key][1] - self.obs_slices[obs_key][key][0]
                        else:
                            input_dim_current += self.obs_slices[obs_key][key][1] - self.obs_slices[obs_key][key][0]
                    
                    input_dim_past = int(input_dim_past/ self.history_num)
                    input_dim_future = int(input_dim_future/ self.future_num)

                    self.storage.register_key(obs_key+'_past', shape=(self.history_num, input_dim_past,), dtype=torch.float)
                    # self.storage.register_key('next_'+obs_key+'_past', shape=(self.history_num, input_dim_past,), dtype=torch.float)
                    self.storage.register_key(obs_key+'_current', shape=(1, input_dim_current,), dtype=torch.float)
                    # self.storage.register_key('next_'+obs_key+'_current', shape=(1, input_dim_current,), dtype=torch.float)
                    self.storage.register_key(obs_key+'_future', shape=(self.future_num, input_dim_future,), dtype=torch.float)
                    # self.storage.register_key('next_'+obs_key+'_future', shape=(self.future_num, input_dim_future,), dtype=torch.float)
        elif self.config.module_dict.actor.type == 'MLP' and self.config.module_dict.critic.type == 'MLP':
            for obs_key, obs_dim in self.algo_obs_dim_dict.items():
                if 'amp' in obs_key:
                    self.storage_amp.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
                    # self.storage.register_key('next_'+obs_key, shape=(obs_dim,), dtype=torch.float)
                else:
                    self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)
                # self.storage.register_key('next_'+obs_key, shape=(obs_dim,), dtype=torch.float)
        
        ## Register others
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('extra_rewards', shape=(1,), dtype=torch.float)
        self.storage.register_key('amp_rewards', shape=(1,), dtype=torch.float)
        self.storage.register_key('dones', shape=(1,), dtype=torch.bool)
        if 'amp' in obs_key and self.num_rew_fn > 1:
            self.storage.register_key('values', shape=(self.num_rew_fn + 1,), dtype=torch.float)
            self.storage.register_key('returns', shape=(self.num_rew_fn + 1,), dtype=torch.float)
            self.storage.register_key('rewards', shape=(self.num_rew_fn + 1,), dtype=torch.float)
        else:
            self.storage.register_key('values', shape=(self.num_rew_fn,), dtype=torch.float)
            self.storage.register_key('returns', shape=(self.num_rew_fn,), dtype=torch.float)
            self.storage.register_key('rewards', shape=(self.num_rew_fn,), dtype=torch.float)
        self.storage.register_key('advantages', shape=(1,), dtype=torch.float)
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float)
        self.storage.register_key('action_mean', shape=(self.num_act,), dtype=torch.float)
        self.storage.register_key('action_sigma', shape=(self.num_act,), dtype=torch.float)
        if 'amp_obs' in self.storage_amp.get_keys():
            self.amp_obs_shape = self.storage_amp.query_key('amp_obs').shape
            # self.storage_amp.register_key('agent_historical_self_obs', shape=(self.amp_obs_shape[2],), dtype=torch.float)
            self.storage_amp.register_key('replay_historical_self_obs', shape=(self.amp_obs_shape[2],), dtype=torch.float)
            self.storage_amp.register_key('expert_historical_self_obs', shape=(self.amp_obs_shape[2],), dtype=torch.float)
    
    def _eval_mode(self):
        super()._eval_mode()
        self.discriminator.eval()

    def _train_mode(self):
        super()._train_mode()
        self.discriminator.train()
    
    def load(self, ckpt_path):
        # import ipdb; ipdb.set_trace()
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            self.discriminator.load_state_dict(loaded_dict["discriminator_model_state_dict"])
            if self.load_optimizer:
                self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
                self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
                self.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
                self.actor_learning_rate = loaded_dict['actor_optimizer_state_dict']['param_groups'][0]['lr']
                self.critic_learning_rate = loaded_dict['critic_optimizer_state_dict']['param_groups'][0]['lr']
                self.discriminator_learning_rate = loaded_dict['discriminator_optimizer_state_dict']['param_groups'][0]['lr']
                self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate, self.discriminator_learning_rate)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rate}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rate}")
                logger.info(f"Discriminator Learning rate: {self.discriminator_learning_rate}")
            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]
    
    def save(self, path, infos=None):
        if self.fabric.global_rank == 0:
            logger.info(f"Saving checkpoint to {path}")
        
        state_dict = {
            'actor_model_state_dict': self.actor.state_dict(),
            'critic_model_state_dict': self.critic.state_dict(),
            'discriminator_model_state_dict': self.discriminator.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        
        # 使用 fabric.save 保存 checkpoint
        self.fabric.save(path, state_dict)
        
        # 同步所有进程
        self.fabric.barrier()
    
    def set_learning_rate(self, actor_learning_rate, critic_learning_rate, discriminator_learning_rate):
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate

    # -----------------------------
    # Experience Buffer and Dataset Processing
    # -----------------------------

    def update_disc_replay_buffer(self, data_dict):
        buf_size = self.amp_replay_buffer.get_buffer_size()
        buf_total_count = len(self.amp_replay_buffer)

        values = list(data_dict.values())
        numel = values[0].shape[0]

        for i in range(1, len(values)):
            assert numel == values[i].shape[0]

        if buf_total_count > buf_size:
            keep_probs = (
                torch.ones(numel, device=self.device)
                * self.config.discriminator_replay_keep_prob
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            for k, v in data_dict.items():
                data_dict[k] = v[keep_mask]

        if numel > buf_size:
            rand_idx = torch.randperm(numel)
            rand_idx = rand_idx[:buf_size]
            for k, v in data_dict.items():
                data_dict[k] = v[rand_idx]

        self.amp_replay_buffer.store(data_dict)
    
    def visualize_3d_obs_points(self, agent_obs, replay_obs, expert_obs, step=0, save_path="./obs_3d_visualization"):
        """
        将三维观测数据平均后,在3D空间中可视化点
        假设 obs_dim = n * 3,每个点有(x, y, z)坐标
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 检查维度
        print(f"\n=== 3D点可视化 (Step {step}) ===")
        print(f"Agent obs shape: {agent_obs.shape}")
        print(f"Replay obs shape: {replay_obs.shape}")
        print(f"Expert obs shape: {expert_obs.shape}")
        
        # 验证obs_dim能被3整除
        obs_dim = agent_obs.shape[-1]
        if obs_dim % 3 != 0:
            print(f"⚠️ 警告: obs_dim={obs_dim} 不能被3整除，无法转换为3D点")
            # 取能整除3的最大值
            obs_dim = (obs_dim // 3) * 3
            print(f"  将使用前{obs_dim}个维度")
        
        n_points = obs_dim // 3
        print(f"将创建 {n_points} 个3D点 (obs_dim={obs_dim}, n_points={n_points})")
        
        def process_obs_to_3d_points(obs_tensor, name):
            """将观测张量转换为3D点"""
            # 1. 平均掉前两个维度 (num_transitions, num_envs)
            # 形状: (num_transitions, num_envs, obs_dim) -> (obs_dim,)
            obs_mean = obs_tensor[..., :obs_dim].mean(dim=(0, 1)).detach().cpu().numpy()
            
            # 2. 重塑为 (n_points, 3)
            points_3d = obs_mean.reshape(-1, 3)
            
            # 3. 统计信息
            print(f"\n{name} 3D点统计:")
            print(f"  点数量: {len(points_3d)}")
            print(f"  X范围: [{points_3d[:, 0].min():.4f}, {points_3d[:, 0].max():.4f}]")
            print(f"  Y范围: [{points_3d[:, 1].min():.4f}, {points_3d[:, 1].max():.4f}]")
            print(f"  Z范围: [{points_3d[:, 2].min():.4f}, {points_3d[:, 2].max():.4f}]")
            print(f"  点范数均值: {np.linalg.norm(points_3d, axis=1).mean():.4f}")
            
            return points_3d
        
        # 处理三个数据集
        agent_points = process_obs_to_3d_points(agent_obs, "Agent")
        replay_points = process_obs_to_3d_points(replay_obs, "Replay")
        expert_points = process_obs_to_3d_points(expert_obs, "Expert")
        
        # 1. 创建3D散点图
        fig = plt.figure(figsize=(15, 10))
        
        # 1.1 主图：三个数据集在一起
        ax1 = fig.add_subplot(231, projection='3d')
        
        # 绘制点
        scatter1 = ax1.scatter(agent_points[:, 0], agent_points[:, 1], agent_points[:, 2], 
                            c='blue', alpha=0.6, s=20, label='Agent', marker='o')
        scatter2 = ax1.scatter(replay_points[:, 0], replay_points[:, 1], replay_points[:, 2], 
                            c='orange', alpha=0.6, s=20, label='Replay', marker='^')
        scatter3 = ax1.scatter(expert_points[:, 0], expert_points[:, 1], expert_points[:, 2], 
                            c='green', alpha=0.6, s=20, label='Expert', marker='s')
        
        ax1.set_title(f'All 3D Points Comparison\n(Step {step})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # 1.2-1.4 分别绘制每个数据集
        views = [(230, 30), (150, 30), (90, 30)]  # 不同的视角
        
        for idx, (name, points, color, ax_pos) in enumerate([
            ('Agent', agent_points, 'blue', 232),
            ('Replay', replay_points, 'orange', 233),
            ('Expert', expert_points, 'green', 234)
        ]):
            ax = fig.add_subplot(ax_pos, projection='3d')
            
            # 绘制散点
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                            c=color, alpha=0.8, s=30)
            
            # 计算质心
            centroid = points.mean(axis=0)
            ax.scatter(*centroid, c='red', s=100, marker='*', label='Centroid')
            
            # 添加从原点到质心的线
            ax.plot([0, centroid[0]], [0, centroid[1]], [0, centroid[2]], 
                    'r--', alpha=0.5)
            
            ax.set_title(f'{name} 3D Points\nCentroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
            # 设置视角
            ax.view_init(*views[idx])
        
        # 1.5 质心对比
        ax5 = fig.add_subplot(235, projection='3d')
        
        centroids = {
            'Agent': agent_points.mean(axis=0),
            'Replay': replay_points.mean(axis=0),
            'Expert': expert_points.mean(axis=0)
        }
        
        for name, centroid in centroids.items():
            color = {'Agent': 'blue', 'Replay': 'orange', 'Expert': 'green'}[name]
            ax5.scatter(*centroid, c=color, s=200, marker='*', label=name, alpha=0.8)
            
            # 添加标签
            ax5.text(centroid[0], centroid[1], centroid[2], 
                    f'{name}\n({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})',
                    fontsize=8)
        
        # 绘制原点
        ax5.scatter(0, 0, 0, c='black', s=50, marker='o', label='Origin')
        
        ax5.set_title('Centroids Comparison')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        ax5.legend()
        
        # 1.6 距离分布直方图
        ax6 = fig.add_subplot(236)
        
        # 计算每个点到原点的距离
        agent_dist = np.linalg.norm(agent_points, axis=1)
        replay_dist = np.linalg.norm(replay_points, axis=1)
        expert_dist = np.linalg.norm(expert_points, axis=1)
        
        ax6.hist(agent_dist, bins=30, alpha=0.5, label='Agent', color='blue', density=True)
        ax6.hist(replay_dist, bins=30, alpha=0.5, label='Replay', color='orange', density=True)
        ax6.hist(expert_dist, bins=30, alpha=0.5, label='Expert', color='green', density=True)
        
        ax6.set_title('Distance from Origin Distribution')
        ax6.set_xlabel('Distance')
        ax6.set_ylabel('Density')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/3d_points_comparison_step_{step}.png', dpi=120, bbox_inches='tight')
        
        # 2. 创建单独的大图，显示点的连接
        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        # 为每个数据集绘制点和连接线
        for points, color, name, marker in [
            (agent_points, 'blue', 'Agent', 'o'),
            (replay_points, 'orange', 'Replay', '^'),
            (expert_points, 'green', 'Expert', 's')
        ]:
            # 绘制点
            ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                    c=color, alpha=0.7, s=40, label=name, marker=marker)
            
            # 按顺序连接点（假设点是有序的）
            if len(points) > 1:
                # 连接线
                for i in range(len(points)-1):
                    ax2.plot([points[i, 0], points[i+1, 0]],
                            [points[i, 1], points[i+1, 1]],
                            [points[i, 2], points[i+1, 2]],
                            c=color, alpha=0.3, linewidth=0.5)
                
                # 连接首尾点形成闭环
                ax2.plot([points[-1, 0], points[0, 0]],
                        [points[-1, 1], points[0, 1]],
                        [points[-1, 2], points[0, 2]],
                        c=color, alpha=0.3, linewidth=0.5, linestyle='--')
        
        ax2.set_title(f'3D Points with Connections (Step {step})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        # 添加坐标轴范围指示
        all_points = np.vstack([agent_points, replay_points, expert_points])
        max_range = np.array([all_points[:, 0].max()-all_points[:, 0].min(),
                            all_points[:, 1].max()-all_points[:, 1].min(),
                            all_points[:, 2].max()-all_points[:, 2].min()]).max() / 2.0
        
        mid_x = (all_points[:, 0].max()+all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max()+all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max()+all_points[:, 2].min()) * 0.5
        
        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/3d_points_connected_step_{step}.png', dpi=120, bbox_inches='tight')
        
        # 3. 创建点的索引图（显示点的顺序）
        fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (points, color, name, ax) in enumerate([
            (agent_points, 'blue', 'Agent', axes[0]),
            (replay_points, 'orange', 'Replay', axes[1]),
            (expert_points, 'green', 'Expert', axes[2])
        ]):
            # 绘制点
            scatter = ax.scatter(range(len(points)), np.linalg.norm(points, axis=1), 
                            c=color, alpha=0.7, s=30)
            
            # 添加点编号
            for i, (x, y) in enumerate(zip(range(len(points)), np.linalg.norm(points, axis=1))):
                ax.text(x, y, str(i), fontsize=6, ha='center', va='bottom')
            
            ax.set_title(f'{name} Point Distances from Origin')
            ax.set_xlabel('Point Index')
            ax.set_ylabel('Distance')
            ax.grid(True, alpha=0.3)
            
            # 显示每个点的坐标
            print(f"\n{name} 前5个点坐标:")
            for i in range(min(5, len(points))):
                print(f"  点{i}: ({points[i, 0]:.4f}, {points[i, 1]:.4f}, {points[i, 2]:.4f})")
        
        plt.suptitle(f'Point Index vs Distance (Step {step})')
        plt.tight_layout()
        plt.savefig(f'{save_path}/point_index_distance_step_{step}.png', dpi=100, bbox_inches='tight')
        
        # 4. 创建坐标分量对比图
        fig4, axes = plt.subplots(3, 3, figsize=(15, 10))
        coord_names = ['X', 'Y', 'Z']
        
        for coord_idx in range(3):  # 遍历X,Y,Z
            for data_idx, (points, name, color) in enumerate([
                (agent_points, 'Agent', 'blue'),
                (replay_points, 'Replay', 'orange'),
                (expert_points, 'Expert', 'green')
            ]):
                ax = axes[coord_idx, data_idx]
                
                # 绘制该坐标分量的值
                values = points[:, coord_idx]
                ax.bar(range(len(values)), values, color=color, alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                
                ax.set_title(f'{name} - {coord_names[coord_idx]} Coordinates')
                ax.set_xlabel('Point Index')
                ax.set_ylabel(f'{coord_names[coord_idx]} Value')
                ax.grid(True, alpha=0.3, axis='y')
                
                # 添加统计信息
                stats_text = f'Mean: {values.mean():.4f}\nStd: {values.std():.4f}'
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Coordinate Components Analysis (Step {step})')
        plt.tight_layout()
        plt.savefig(f'{save_path}/coordinate_components_step_{step}.png', dpi=100, bbox_inches='tight')
        
        plt.close('all')
        
        print(f"\n3D可视化已保存到: {save_path}/")
        print(f"生成的文件:")
        print(f"  - 3d_points_comparison_step_{step}.png: 6个子图综合对比")
        print(f"  - 3d_points_connected_step_{step}.png: 带连接线的大图")
        print(f"  - point_index_distance_step_{step}.png: 点索引与距离")
        print(f"  - coordinate_components_step_{step}.png: 坐标分量分析")
        
        # 返回处理后的点用于进一步分析
        return {
            'step': step,
            'agent_points': agent_points,
            'replay_points': replay_points,
            'expert_points': expert_points,
            'n_points': n_points
        }
    
    def visualize_obs(self, agent_obs, replay_obs, expert_obs, step=0, save_path="./obs_visualization"):
        """
        将单维观测数据平均后,在1D/2D空间中可视化折线图
        假设输入是单维数据
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 检查维度
        print(f"\n=== 单变量可视化 (Step {step}) ===")
        print(f"Agent obs shape: {agent_obs.shape}")
        print(f"Replay obs shape: {replay_obs.shape}")
        print(f"Expert obs shape: {expert_obs.shape}")
        
        # 获取数据维度
        obs_dim = agent_obs.shape[-1]
        print(f"观测维度: {obs_dim}")
        
        def process_obs_to_1d_data(obs_tensor, name):
            """将观测张量转换为1D数据"""
            # 1. 平均掉前两个维度 (num_transitions, num_envs)
            # 形状: (num_transitions, num_envs, obs_dim) -> (obs_dim,)
            obs_mean = obs_tensor[..., :obs_dim].mean(dim=(0, 1)).detach().cpu().numpy()
            
            # 2. 统计信息
            print(f"\n{name} 数据统计:")
            print(f"  数据长度: {len(obs_mean)}")
            print(f"  数值范围: [{obs_mean.min():.4f}, {obs_mean.max():.4f}]")
            print(f"  均值: {obs_mean.mean():.4f}, 标准差: {obs_mean.std():.4f}")
            print(f"  中位数: {np.median(obs_mean):.4f}")
            
            return obs_mean
        
        # 处理三个数据集
        agent_data = process_obs_to_1d_data(agent_obs, "Agent")
        replay_data = process_obs_to_1d_data(replay_obs, "Replay")
        expert_data = process_obs_to_1d_data(expert_obs, "Expert")
        
        # 创建时间/索引轴
        x_axis = np.arange(len(agent_data))
        
        # 1. 创建综合对比图
        fig = plt.figure(figsize=(18, 12))
        
        # 1.1 三个数据集叠加对比
        ax1 = fig.add_subplot(231)
        ax1.plot(x_axis, agent_data, 'b-', linewidth=1.5, alpha=0.8, label='Agent')
        ax1.plot(x_axis, replay_data, 'orange', linewidth=1.5, alpha=0.8, label='Replay', linestyle='--')
        ax1.plot(x_axis, expert_data, 'g-', linewidth=1.5, alpha=0.8, label='Expert', linestyle='-.')
        ax1.set_title(f'Data Comparison (Step {step})')
        ax1.set_xlabel('Dimension Index')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 1.2 单独显示Agent
        ax2 = fig.add_subplot(232)
        ax2.plot(x_axis, agent_data, 'b-', linewidth=2, alpha=0.8)
        ax2.fill_between(x_axis, agent_data, alpha=0.3, color='blue')
        ax2.set_title(f'Agent Data\nMean: {agent_data.mean():.4f}, Std: {agent_data.std():.4f}')
        ax2.set_xlabel('Dimension Index')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        # 1.3 单独显示Replay
        ax3 = fig.add_subplot(233)
        ax3.plot(x_axis, replay_data, 'orange', linewidth=2, alpha=0.8)
        ax3.fill_between(x_axis, replay_data, alpha=0.3, color='orange')
        ax3.set_title(f'Replay Data\nMean: {replay_data.mean():.4f}, Std: {replay_data.std():.4f}')
        ax3.set_xlabel('Dimension Index')
        ax3.set_ylabel('Value')
        ax3.grid(True, alpha=0.3)
        
        # 1.4 单独显示Expert
        ax4 = fig.add_subplot(234)
        ax4.plot(x_axis, expert_data, 'g-', linewidth=2, alpha=0.8)
        ax4.fill_between(x_axis, expert_data, alpha=0.3, color='green')
        ax4.set_title(f'Expert Data\nMean: {expert_data.mean():.4f}, Std: {expert_data.std():.4f}')
        ax4.set_xlabel('Dimension Index')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
        
        # 1.5 误差图（与Expert的差异）
        ax5 = fig.add_subplot(235)
        agent_error = agent_data - expert_data
        replay_error = replay_data - expert_data
        
        ax5.plot(x_axis, agent_error, 'b-', linewidth=1.5, alpha=0.7, label='Agent - Expert')
        ax5.plot(x_axis, replay_error, 'orange', linewidth=1.5, alpha=0.7, label='Replay - Expert', linestyle='--')
        ax5.axhline(y=0, color='r', linestyle='-', linewidth=1, alpha=0.5)
        ax5.set_title(f'Error Compared to Expert\nMAE(Agent): {np.abs(agent_error).mean():.4f}, MAE(Replay): {np.abs(replay_error).mean():.4f}')
        ax5.set_xlabel('Dimension Index')
        ax5.set_ylabel('Error')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 1.6 统计分布直方图
        ax6 = fig.add_subplot(236)
        ax6.hist(agent_data, bins=30, alpha=0.5, label='Agent', color='blue', density=True)
        ax6.hist(replay_data, bins=30, alpha=0.5, label='Replay', color='orange', density=True)
        ax6.hist(expert_data, bins=30, alpha=0.5, label='Expert', color='green', density=True)
        ax6.set_title('Value Distribution')
        ax6.set_xlabel('Value')
        ax6.set_ylabel('Density')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Single Variable Visualization (Step {step})', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{save_path}/data_comparison_step_{step}.png', dpi=120, bbox_inches='tight')
        
        # 2. 创建滑动窗口平滑对比
        fig2, (ax21, ax22) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 平滑窗口大小
        window_size = min(20, len(agent_data) // 10)
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            
            agent_smooth = np.convolve(agent_data, kernel, mode='valid')
            replay_smooth = np.convolve(replay_data, kernel, mode='valid')
            expert_smooth = np.convolve(expert_data, kernel, mode='valid')
            x_smooth = np.arange(window_size-1, len(agent_data))
            
            ax21.plot(x_smooth, agent_smooth, 'b-', linewidth=2, alpha=0.8, label='Agent (smoothed)')
            ax21.plot(x_smooth, replay_smooth, 'orange', linewidth=2, alpha=0.8, label='Replay (smoothed)', linestyle='--')
            ax21.plot(x_smooth, expert_smooth, 'g-', linewidth=2, alpha=0.8, label='Expert (smoothed)', linestyle='-.')
            ax21.set_title(f'Smoothed Data (window={window_size})')
            ax21.set_xlabel('Dimension Index')
            ax21.set_ylabel('Value')
            ax21.legend()
            ax21.grid(True, alpha=0.3)
        
        # 2.2 累积分布函数比较
        ax22.hist(agent_data, bins=50, histtype='step', cumulative=True, 
                density=True, label='Agent', color='blue', linewidth=2)
        ax22.hist(replay_data, bins=50, histtype='step', cumulative=True, 
                density=True, label='Replay', color='orange', linewidth=2, linestyle='--')
        ax22.hist(expert_data, bins=50, histtype='step', cumulative=True, 
                density=True, label='Expert', color='green', linewidth=2, linestyle='-.')
        ax22.set_title('Cumulative Distribution Function (CDF)')
        ax22.set_xlabel('Value')
        ax22.set_ylabel('Cumulative Probability')
        ax22.legend()
        ax22.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/smoothed_cdf_step_{step}.png', dpi=120, bbox_inches='tight')
        
        # 3. 创建分位数-分位数图
        fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Agent vs Expert QQ图
        ax31.scatter(np.sort(expert_data), np.sort(agent_data), alpha=0.6, s=20, c='blue')
        min_val = min(expert_data.min(), agent_data.min())
        max_val = max(expert_data.max(), agent_data.max())
        ax31.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        ax31.set_title('Agent vs Expert QQ Plot')
        ax31.set_xlabel('Expert Quantiles')
        ax31.set_ylabel('Agent Quantiles')
        ax31.legend()
        ax31.grid(True, alpha=0.3)
        
        # Replay vs Expert QQ图
        ax32.scatter(np.sort(expert_data), np.sort(replay_data), alpha=0.6, s=20, c='orange')
        ax32.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        ax32.set_title('Replay vs Expert QQ Plot')
        ax32.set_xlabel('Expert Quantiles')
        ax32.set_ylabel('Replay Quantiles')
        ax32.legend()
        ax32.grid(True, alpha=0.3)
        
        plt.suptitle(f'Quantile-Quantile Plots (Step {step})')
        plt.tight_layout()
        plt.savefig(f'{save_path}/qq_plots_step_{step}.png', dpi=100, bbox_inches='tight')
        
        # 4. 创建统计摘要图
        fig4, ((ax41, ax42), (ax43, ax44)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 4.1 箱线图比较
        box_data = [agent_data, replay_data, expert_data]
        box_labels = ['Agent', 'Replay', 'Expert']
        box_colors = ['blue', 'orange', 'green']
        
        bp = ax41.boxplot(box_data, labels=box_labels, patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax41.set_title('Box Plot Comparison')
        ax41.set_ylabel('Value')
        ax41.grid(True, alpha=0.3, axis='y')
        
        # 4.2 标准差对比
        stds = [agent_data.std(), replay_data.std(), expert_data.std()]
        ax42.bar(box_labels, stds, color=box_colors, alpha=0.7)
        ax42.set_title('Standard Deviation Comparison')
        ax42.set_ylabel('Standard Deviation')
        for i, v in enumerate(stds):
            ax42.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        ax42.grid(True, alpha=0.3, axis='y')
        
        # 4.3 范围对比
        ranges = [agent_data.max()-agent_data.min(), 
                replay_data.max()-replay_data.min(), 
                expert_data.max()-expert_data.min()]
        ax43.bar(box_labels, ranges, color=box_colors, alpha=0.7)
        ax43.set_title('Range Comparison (max-min)')
        ax43.set_ylabel('Range')
        for i, v in enumerate(ranges):
            ax43.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        ax43.grid(True, alpha=0.3, axis='y')
        
        # 4.4 与Expert的均方误差
        mse_agent = np.mean((agent_data - expert_data) ** 2)
        mse_replay = np.mean((replay_data - expert_data) ** 2)
        mses = [mse_agent, mse_replay, 0]
        ax44.bar(['Agent', 'Replay', 'Expert'], mses, color=['blue', 'orange', 'green'], alpha=0.7)
        ax44.set_title('Mean Squared Error vs Expert')
        ax44.set_ylabel('MSE')
        for i, v in enumerate(mses[:2]):  # 只标注前两个
            ax44.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        ax44.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Statistical Summary (Step {step})')
        plt.tight_layout()
        plt.savefig(f'{save_path}/statistical_summary_step_{step}.png', dpi=100, bbox_inches='tight')
        
        plt.close('all')
        
        print(f"\n可视化已保存到: {save_path}/")
        print(f"生成的文件:")
        print(f"  - data_comparison_step_{step}.png: 6个子图综合对比")
        print(f"  - smoothed_cdf_step_{step}.png: 平滑数据和累积分布")
        print(f"  - qq_plots_step_{step}.png: 分位数-分位数图")
        print(f"  - statistical_summary_step_{step}.png: 统计摘要")
        
        # 返回处理后的数据用于进一步分析
        return {
            'step': step,
            'agent_data': agent_data,
            'replay_data': replay_data,
            'expert_data': expert_data,
            'stats': {
                'agent': {
                    'mean': float(agent_data.mean()),
                    'std': float(agent_data.std()),
                    'min': float(agent_data.min()),
                    'max': float(agent_data.max()),
                    'mse_vs_expert': float(np.mean((agent_data - expert_data) ** 2))
                },
                'replay': {
                    'mean': float(replay_data.mean()),
                    'std': float(replay_data.std()),
                    'min': float(replay_data.min()),
                    'max': float(replay_data.max()),
                    'mse_vs_expert': float(np.mean((replay_data - expert_data) ** 2))
                },
                'expert': {
                    'mean': float(expert_data.mean()),
                    'std': float(expert_data.std()),
                    'min': float(expert_data.min()),
                    'max': float(expert_data.max())
                }
            }
        }

    @torch.no_grad()
    def process_amp_obs(self):
        # Read historical observations from the AMP storage (separate buffer)
        historical_self_obs = self.storage_amp.query_key("amp_obs")

        num_samples = historical_self_obs.shape[0] * historical_self_obs.shape[1]

        if len(self.amp_replay_buffer) == 0:
            replay_historical_self_obs = historical_self_obs
        else:
            replay_dict = self.amp_replay_buffer.sample(num_samples)
            replay_historical_self_obs = replay_dict["historical_self_obs"]
            replay_historical_self_obs = replay_historical_self_obs.view(historical_self_obs.shape[0], historical_self_obs.shape[1], -1)

        expert_historical_self_obs = self.get_expert_historical_self_obs(num_samples)
        expert_historical_self_obs = expert_historical_self_obs.view(historical_self_obs.shape[0], historical_self_obs.shape[1], -1)

        # self.storage_amp.batch_update_data("agent_historical_self_obs", historical_self_obs)
        self.storage_amp.batch_update_data("replay_historical_self_obs", replay_historical_self_obs)
        self.storage_amp.batch_update_data("expert_historical_self_obs", expert_historical_self_obs)

        # result = self.visualize_3d_obs_points(
        #     agent_obs=historical_self_obs[...,:10],
        #     replay_obs=replay_historical_self_obs[...,:10],
        #     expert_obs=expert_historical_self_obs[...,:10],
        #     # expert_obs=self.ref_state.reshape(24,8,870)[...,:870],
        #     step=self.global_step if hasattr(self, 'global_step') else 0
        # )

        # result = self.visualize_obs(
        #     agent_obs=historical_self_obs[...,10:240],
        #     replay_obs=replay_historical_self_obs[...,10:240],
        #     expert_obs=expert_historical_self_obs[...,10:240],
        #     # expert_obs=self.ref_state.reshape(24,8,870)[...,:870],
        #     step=self.global_step if hasattr(self, 'global_step') else 0
        # )

        historical_self_obs = swap_and_flatten01(
            historical_self_obs
        )
        self.update_disc_replay_buffer({"historical_self_obs": historical_self_obs})

    def get_expert_historical_self_obs(self, num_samples: int):
        motion_ids = self.env._motion_lib.sample_motions(num_samples, random_sample=True)
        num_steps = self.env.config.obs.history_len_amp

        dt = self.env.dt
        truncate_time = dt * (num_steps - 1) + 1

        # Since negative times are added to these values in build_historical_self_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip].
        motion_times0 = self.env._motion_lib.sample_time(
            motion_ids, truncate_time=truncate_time
        )
        motion_times0 = motion_times0 + truncate_time
        # motion_times0 = truncate_time * torch.ones_like(motion_ids, dtype=torch.float)

        obs = self.build_self_obs_demo(
            motion_ids, motion_times0, num_steps
        ).clone()
        return obs.view(num_samples, -1)

    def build_self_obs_demo(
        self, motion_ids: Tensor, motion_times0: Tensor, num_steps: int
    ):
        dt = self.env.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, num_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, num_steps, device=self.env.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)

        lengths = self.env._motion_lib.get_motion_length(motion_ids)

        motion_times = motion_times.view(-1).clamp(max=lengths).clamp(min=0)

        ref_state = self.env._motion_lib.get_motion_state(motion_ids, motion_times)
        # self.ref_state = ref_state['rg_pos_t'].reshape(num_steps, motion_times0.shape[0], -1, 3).transpose(0, 1).reshape(motion_times0.shape[0], -1)
        # print(f"Expert ref_state shape: {self.ref_state.shape}")
        
        # Prepare dof_pos: shape is [num_steps * batch_size, num_dofs]
        dof_pos_ref = ref_state['dof_pos'].reshape(num_steps, motion_times0.shape[0], -1).transpose(0, 1)
        
        # Get default_dof_pos from env if available
        default_dof_pos = self.env.default_dof_pos if hasattr(self.env, 'default_dof_pos') else None
        default_dof_pos = default_dof_pos.clone().detach()
        
        obs_demo = compute_humanoid_observations_max(
            ref_state['rg_pos_t'].reshape(num_steps, motion_times0.shape[0], -1, 3).transpose(0, 1),
            ref_state['rg_rot_t'].reshape(num_steps, motion_times0.shape[0], -1, 4).transpose(0, 1),
            ref_state['body_vel_t'].reshape(num_steps, motion_times0.shape[0], -1, 3).transpose(0, 1),
            ref_state['body_ang_vel_t'].reshape(num_steps, motion_times0.shape[0], -1, 3).transpose(0, 1),
            True,
            dof_pos_ref,
            default_dof_pos
        )
        return obs_demo


    # -----------------------------
    # Optimization
    # -----------------------------
    def extra_optimization_steps(self, policy_state_dict, loss_dict):
        discriminator_loss, loss_dict = self.discriminator_step(
            policy_state_dict, loss_dict
        )
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(discriminator_loss)
        self.fabric.clip_gradients(
            self.discriminator, 
            self.discriminator_optimizer, 
            max_norm=self.max_grad_norm,
            error_if_nonfinite=False
        )
        self.discriminator_optimizer.step()

        return loss_dict
    

    def discriminator_step(self, policy_state_dict, loss_dict):
        agent_obs = policy_state_dict["amp_obs"][
            : self.config.discriminator_batch_size
        ]
        replay_obs = policy_state_dict["replay_historical_self_obs"][
            : self.config.discriminator_batch_size
        ]
        expert_obs = policy_state_dict["expert_historical_self_obs"][
            : self.config.discriminator_batch_size
        ]
        # print(f"检查输入数据:")
        # print(f"  agent_obs: shape={agent_obs.shape}, "
        #     f"NaN={torch.isnan(agent_obs).any().item()}, "
        #     f"Inf={torch.isinf(agent_obs).any().item()}, "
        #     f"mean={agent_obs.mean().item():.6f}, "
        #     f"std={agent_obs.std().item():.6f}")
        # print(f"  expert_obs: shape={expert_obs.shape}, "
        #     f"NaN={torch.isnan(expert_obs).any().item()}, "
        #     f"Inf={torch.isinf(expert_obs).any().item()}, "
        #     f"mean={expert_obs.mean().item():.6f}, "
        #     f"std={expert_obs.std().item():.6f}")

        agent_obs = torch.nan_to_num(agent_obs, nan=0.0, posinf=1.0, neginf=-1.0)  # 新增
        replay_obs = torch.nan_to_num(replay_obs, nan=0.0, posinf=1.0, neginf=-1.0)  # 新增
        expert_obs = torch.nan_to_num(expert_obs, nan=0.0, posinf=1.0, neginf=-1.0)  # 新增

        combined_obs = torch.cat([agent_obs, expert_obs], dim=0)
        combined_obs.requires_grad_(True)

        combined_dict = self.discriminator.compute_logits(
            {"historical_self_obs": combined_obs}, return_norm_obs=True
        )
        combined_logits = combined_dict["outs"]
        combined_norm_obs = combined_dict["norm_historical_self_obs"]

        replay_logits = self.discriminator.compute_logits(
            {"historical_self_obs": replay_obs}
        )

        agent_logits = combined_logits[: self.config.discriminator_batch_size]
        expert_logits = combined_logits[self.config.discriminator_batch_size :]

        # print(f"logits统计:")
        # print(f"  agent_logits: mean={agent_logits.mean().item():.6f}, "
        #     f"std={agent_logits.std().item():.6f}, "
        #     f"max={agent_logits.max().item():.6f}, "
        #     f"min={agent_logits.min().item():.6f}")
        # print(f"  expert_logits: mean={expert_logits.mean().item():.6f}, "
        #     f"std={expert_logits.std().item():.6f}, "
        #     f"max={expert_logits.max().item():.6f}, "
        #     f"min={expert_logits.min().item():.6f}")
        # print(f"  replay_logits: mean={replay_logits.mean().item():.6f}, "
        #     f"std={replay_logits.std().item():.6f}")
        
        expert_loss = -torch.nn.functional.logsigmoid(expert_logits).mean()
        unlabeled_loss = torch.nn.functional.softplus(agent_logits).mean()
        replay_loss = torch.nn.functional.softplus(replay_logits).mean()

        neg_loss = 0.5 * (unlabeled_loss + replay_loss)
        class_loss = 0.5 * (expert_loss + neg_loss)
        # print(f"neg loss: {neg_loss.mean().item():.6f}, "
        # f"class_loss: {class_loss.mean().item():.6f}, "
        # f"expert_loss: {expert_loss.mean().item():.6f}, "
        # f"unlabeled_loss: {unlabeled_loss.mean().item():.6f}, "
        # f"replay_loss: {replay_loss.mean().item():.6f}")
        # disc_grad = torch.autograd.grad(
        #     combined_logits,
        #     combined_norm_obs,
        #     grad_outputs=torch.ones_like(combined_logits),
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True,
        # )[0]

        disc_grad = self.compute_gradient_penalty(combined_logits, combined_norm_obs)
        # print("dic_grad shape:", disc_grad.shape)
        disc_grad_norm = torch.norm(disc_grad, dim=-1)
        disc_grad_penalty = torch.mean(disc_grad_norm)
        grad_loss: Tensor = self.config.discriminator_grad_penalty * disc_grad_penalty
        if disc_grad_penalty > 50:
            self.discriminator_learning_rate = max(5e-7, self.discriminator_learning_rate / 1.5)
        elif disc_grad_penalty < 0.1:
            self.discriminator_learning_rate = min(1e-3, self.discriminator_learning_rate * 1.5) 

        for param_group in self.discriminator_optimizer.param_groups:
            param_group['lr'] = self.discriminator_learning_rate

        if self.config.discriminator_weight_decay > 0:
            all_weight_params = self.discriminator.all_discriminator_weights()
            total: Tensor = sum([p.pow(2).sum() for p in all_weight_params])
            weight_decay_loss: Tensor = total * self.config.discriminator_weight_decay
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)
            total = torch.tensor(0.0, device=self.device)

        if self.config.discriminator_logit_weight_decay > 0:
            logit_params = self.discriminator.logit_weights()
            logit_total = sum([p.pow(2).sum() for p in logit_params])
            logit_weight_decay_loss: Tensor = (
                logit_total * self.config.discriminator_logit_weight_decay
            )
        else:
            logit_weight_decay_loss = torch.tensor(0.0, device=self.device)
            logit_total = torch.tensor(0.0, device=self.device)

        loss = grad_loss + class_loss + weight_decay_loss + logit_weight_decay_loss

        with torch.no_grad():
            pos_acc = self.compute_pos_acc(expert_logits)
            agent_acc = self.compute_neg_acc(agent_logits)
            replay_acc = self.compute_neg_acc(replay_logits)
            neg_acc = 0.5 * (agent_acc + replay_acc)
            loss_dict["losses/discriminator_loss"] += loss.detach()
            loss_dict["discriminator/pos_acc"]+= pos_acc.detach()
            loss_dict["discriminator/agent_acc"]+= agent_acc.detach()
            loss_dict["discriminator/replay_acc"]+= replay_acc.detach()
            loss_dict["discriminator/neg_acc"]+= neg_acc.detach()
            loss_dict["discriminator/grad_penalty"]+= disc_grad_penalty.detach()
            loss_dict["discriminator/grad_loss"]+= grad_loss.detach()
            loss_dict["discriminator/grad_norm"]+= disc_grad_norm.mean().detach()
            loss_dict["discriminator/class_loss"]+= class_loss.detach()
            loss_dict["discriminator/l2_logit_total"]+= logit_total.detach()
            loss_dict["discriminator/l2_logit_loss"]+= logit_weight_decay_loss.detach()
            loss_dict["discriminator/l2_total"]+= total.detach()
            loss_dict["discriminator/l2_loss"]+= weight_decay_loss.detach()
            loss_dict["discriminator/expert_logit_mean"]+= expert_logits.detach().mean()
            loss_dict["discriminator/agent_logit_mean"]+= agent_logits.detach().mean()
            loss_dict["discriminator/replay_logit_mean"]+= replay_logits.detach().mean()
            loss_dict["discriminator/negative_logit_mean"] += 0.5 * (
                agent_logits.detach().mean()
                + replay_logits.detach().mean()
            )

        return loss, loss_dict

    def compute_gradient_penalty(self, logits, norm_obs):
        if not norm_obs.requires_grad:
            norm_obs = norm_obs.requires_grad_(True)
        
        norm_obs.retain_grad()
        
        # 计算虚拟损失
        # print(f"logits stats: mean={logits.mean().item():.6f}, "
        # f"std={logits.std().item():.6f}, "
        # f"max={logits.max().item():.6f}, "
        # f"min={logits.min().item():.6f}")
    
        dummy_loss = logits.mean()
        
        # 检查是否有fabric
        if hasattr(self, 'fabric') and self.fabric is not None:
            # 使用fabric方式
            self.discriminator.zero_grad()
            if norm_obs.grad is not None:
                norm_obs.grad.zero_()
            
            self.fabric.backward(dummy_loss, retain_graph=True)
            grad = norm_obs.grad.clone() if norm_obs.grad is not None else torch.zeros_like(norm_obs)
            grad = torch.clamp(grad, -1e1, 1e1)  # 防止梯度爆炸
            # print(f"Gradient stats: mean={grad.mean().item():.6f}, "
            # f"std={grad.std().item():.6f}, "
            # f"max={grad.max().item():.6f}, ")
            # 清理梯度
            self.discriminator.zero_grad()
        else:
            # 标准PyTorch方式
            grad = torch.autograd.grad(
                dummy_loss,
                norm_obs,
                create_graph=True,
                retain_graph=True,
            )[0]
        
        return grad

    # -----------------------------
    # Discriminator Metrics and Utility
    # -----------------------------
    @staticmethod
    def compute_pos_acc(positive_logit: Tensor) -> Tensor:
        return (positive_logit > 0).float().mean()

    @staticmethod
    def compute_neg_acc(negative_logit: Tensor) -> Tensor:
        return (negative_logit < 0).float().mean()
    


    # -----------------------------
    # Termination and Logging
    # -----------------------------
    def terminate_early(self):
        self._should_stop = True

    
    # @torch.no_grad()
    # def calculate_extra_reward(self):
    #     rew = torch.zeros(self.num_steps_per_env, self.num_envs, device=self.device)

    #     historical_self_obs = self.storage.query_key("amp_obs")
    #     amp_r = self.discriminator.compute_reward(
    #         {
    #             "historical_self_obs": historical_self_obs.view(
    #                 self.num_envs * self.num_steps_per_env, -1
    #             )
    #         }
    #     ).view(self.num_steps_per_env, self.num_envs)

    #     self.storage.batch_update_data("amp_rewards", amp_r.unsqueeze(-1))

    #     extra_reward = amp_r * self.config.discriminator_reward_w + rew
    #     return extra_reward

    @torch.no_grad()
    def calculate_extra_reward(self, obs_dict):
        historical_self_obs = obs_dict["amp_obs"]
        amp_r = self.discriminator.compute_reward(
            {
                "historical_self_obs": historical_self_obs
            }
        ).view(self.num_envs, 1)

        self.storage.update_key("amp_rewards", amp_r)

        extra_reward = amp_r * self.config.discriminator_reward_w
        return extra_reward

    # Override rollout step to also write historical_self_obs into amp_storage
    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                ## Append states to storage (main storage and amp storage for historical obs)
                for obs_key in obs_dict.keys():
                    if 'amp_obs' in obs_key:
                        self.storage_amp.update_key(obs_key, obs_dict[obs_key])
                    else:
                        self.storage.update_key(obs_key, obs_dict[obs_key])

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])
                actions = policy_state_dict["actions"]
                actor_state = {}
                actor_state["actions"] = actions
                extra_rewards = self.calculate_extra_reward(obs_dict)
                extra_rewards = extra_rewards.to(self.device)
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                    # self.storage.update_key('next_'+obs_key, obs_dict[obs_key])
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])
                if self.env.num_rew_fn > 1:
                    rewards_stored = torch.cat([rewards.clone().reshape(self.env.num_envs, self.env.num_rew_fn), extra_rewards.clone().reshape(self.env.num_envs, 1)], dim=-1)
                else:
                    rewards_stored = rewards.clone().reshape(self.env.num_envs, 1) + extra_rewards.clone().reshape(self.env.num_envs, 1)

                if 'time_outs' in infos:
                    rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                assert len(rewards_stored.shape) == 2
                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                # increment main storage step counter (ExperienceBuffer uses explicit indices)
                self.storage.increment_step()

                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards.view(self.env.num_envs, self.env.num_rew_fn).sum(dim=-1) + extra_rewards.view(self.env.num_envs)
                    self.cur_extra_reward_sum += extra_rewards.view(self.env.num_envs)
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.extrarewbuffer.extend(self.cur_extra_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_extra_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            # prepare data for training
            # print(self.storage.get_keys())
            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            # extra_rewards = self.calculate_extra_reward()
            # self.storage.batch_update_data('extra_rewards', extra_rewards.unsqueeze(-1))
            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(values=self.storage.query_key('values'),
                dones=self.storage.query_key('dones'),
                rewards=self.storage.query_key('rewards'))
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict
    
    def _compute_returns(self, last_obs_dict, policy_state_dict):
        """Compute the returns and advantages for the given policy state.
        This function calculates the returns and advantages for each step in the 
        environment based on the provided observations and policy state. It uses 
        Generalized Advantage Estimation (GAE) to compute the advantages, which 
        helps in reducing the variance of the policy gradient estimates.
        Args:
            last_obs_dict (dict): The last observation dictionary containing the 
                      final state of the environment.
            policy_state_dict (dict): A dictionary containing the policy state 
                          information, including 'values', 'dones', 
                          and 'rewards'.
        Returns:
            tuple: A tuple containing:
            - returns (torch.Tensor): The computed returns for each step.
            - advantages (torch.Tensor): The normalized advantages for each step.
        """
        if "critic_obs" in last_obs_dict:
            last_values = self.critic.evaluate(last_obs_dict["critic_obs"]).detach()
        else:
            # obs_dict_transformer = {}
            # obs_dict_transformer["past"] = last_obs_dict["critic_obs_past"]
            # obs_dict_transformer["current"] = last_obs_dict["critic_obs_current"]
            # obs_dict_transformer["future"] = last_obs_dict["critic_obs_future"]
            last_values = self.critic.evaluate(last_obs_dict).detach()
        
        values = policy_state_dict['values']
        dones = policy_state_dict['dones']
        rewards = policy_state_dict['rewards']
        
        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        
        returns = torch.zeros_like(values)
        # advantages = torch.zeros_like(dones)  # not vec, it must be a scalar
        
        num_steps = returns.shape[0]
        advantage = 0
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        tot_advantages = returns - values
        aggr_tot_advantages = tot_advantages.sum(dim=-1)
        advantages = (aggr_tot_advantages - aggr_tot_advantages.mean()) / (aggr_tot_advantages.std() + 1e-8)
        return returns, advantages.unsqueeze(-1)
    
    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()
        self.process_amp_obs()
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        generator_amp = self.storage_amp.mini_batch_generator(self.config.discriminator_mini_batches, self.config.discriminator_learning_epochs)

        for policy_state_dict in generator:
            # Move everything to the device
            for policy_state_key in policy_state_dict.keys():
                policy_state_dict[policy_state_key] = policy_state_dict[policy_state_key].to(self.device)
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)
        
        for policy_state_dict_amp in generator_amp:
            # Move everything to the device
            for policy_state_key_amp in policy_state_dict_amp.keys():
                policy_state_dict_amp[policy_state_key_amp] = policy_state_dict_amp[policy_state_key_amp].to(self.device)
            loss_dict = self.extra_optimization_steps(policy_state_dict_amp, loss_dict)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        num_updates_amp = self.config.discriminator_learning_epochs * self.config.discriminator_mini_batches
        for key in loss_dict.keys():
            if 'discriminator' in key:
                loss_dict[key] /= num_updates_amp
            else:
                loss_dict[key] /= num_updates
        self.storage.clear()
        self.storage_amp.clear()
        return loss_dict
    
    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        loss_dict['Value'] = 0
        loss_dict['Surrogate'] = 0
        loss_dict['Entropy'] = 0
        loss_dict['L2C2_Value'] = 0
        loss_dict['L2C2_Policy'] = 0
        loss_dict["losses/discriminator_loss"] = 0
        loss_dict["discriminator/pos_acc"]= 0
        loss_dict["discriminator/agent_acc"]= 0
        loss_dict["discriminator/replay_acc"]= 0
        loss_dict["discriminator/neg_acc"]= 0
        loss_dict["discriminator/grad_penalty"]= 0
        loss_dict["discriminator/grad_loss"]= 0
        loss_dict["discriminator/grad_norm"]= 0
        loss_dict["discriminator/class_loss"]= 0
        loss_dict["discriminator/l2_logit_total"]= 0
        loss_dict["discriminator/l2_logit_loss"]= 0
        loss_dict["discriminator/l2_total"]= 0
        loss_dict["discriminator/l2_loss"]= 0
        loss_dict["discriminator/expert_logit_mean"]= 0
        loss_dict["discriminator/agent_logit_mean"]= 0
        loss_dict["discriminator/replay_logit_mean"]= 0
        loss_dict["discriminator/negative_logit_mean"] = 0
        return loss_dict

    def learn(self):
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
        
        self._train_mode()

        num_learning_iterations = self.num_learning_iterations

        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        # do not use track, because it will confict with motion loading bar
        # for it in track(range(self.current_learning_iteration, tot_iter), description="Learning Iterations"):
        for it in range(self.current_learning_iteration, tot_iter):
            self.start_time = time.time()

            obs_dict =self._rollout_step(obs_dict)

            loss_dict = self._training_step()

            self.stop_time = time.time()
            self.learn_time = self.stop_time - self.start_time

            # Logging
            log_dict = {
                'it': it,
                'loss_dict': loss_dict,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
                'ep_infos': self.ep_infos, # len(ep_infos) = 24 = rollout steps
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'extrarewbuffer': self.extrarewbuffer,
                'num_learning_iterations': num_learning_iterations,
            }
            self._post_epoch_logging(log_dict)
            if it % self.save_interval == 0:
                self.current_learning_iteration = it
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            self.ep_infos.clear()
        
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
    
    def _post_epoch_logging(self, log_dict, width=80, pad=40):
        # Update total timesteps and total time
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs * self.fabric.world_size
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']
        
        if log_dict['it'] % self.logging_interval != 0:  # Check report frequency
            return
        
        # 准备要记录到 Fabric 的指标字典
        fabric_metrics = {}

        # Closure functions to generate log strings
        def generate_computation_log():
            # Calculate mean standard deviation and frames per second (FPS)
            mean_std = self.actor.std.mean()
            fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
            str = f" \033[1m Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "
            
            return (f"""{str.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std:>10.4f}\n""")

        def generate_reward_length_log():
            # Generate log for mean reward and mean episode length
            reward_length_string = ""
            
            
            if len(log_dict['rewbuffer']) > 0:
                mean_reward = statistics.mean(log_dict['rewbuffer'])
                mean_length = statistics.mean(log_dict['lenbuffer'])
                mean_extra_reward = statistics.mean(log_dict['extrarewbuffer'])
                reward_length_string += (f"""{'Mean reward:':>{pad}} {mean_reward:>10.4f}\n"""
                                         f"""{'Mean episode length:':>{pad}} {mean_length:>10.4f}\n"""
                                         f"""{'Mean extra reward:':>{pad}} {mean_extra_reward:>10.4f}\n""")
                
                # 收集指标到 fabric_metrics
                fabric_metrics['Train/mean_reward'] = mean_reward
                fabric_metrics['Train/mean_episode_length'] = mean_length
                fabric_metrics['Train/mean_extra_reward'] = mean_extra_reward
                
                # TensorBoard 记录 (只在主进程)
                if self.fabric.global_rank == 0:
                    self.writer.add_scalar('Train/mean_reward', mean_reward, log_dict['it'])
                    self.writer.add_scalar('Train/mean_episode_length', mean_length, log_dict['it'])
                    self.writer.add_scalar('Train/mean_extra_reward', mean_extra_reward, log_dict['it'])
            return reward_length_string

        def generate_env_log():
            # Generate log for environment metrics
            env_log_string = ""
            env_log_dict = self.episode_env_tensors.mean_and_clear()
            env_log_dict = {f"{k}": v for k, v in env_log_dict.items()}
            
            for k, v in env_log_dict.items():
                entry = f"{f'{k}:':>{pad}} {v:>10.4f}"
                env_log_string += f"{entry}\n"
                # 收集到 fabric_metrics
                fabric_metrics[f'Env/{k}'] = v
                
                # TensorBoard 记录 (只在主进程)
                if self.fabric.global_rank == 0:
                    self.writer.add_scalar('Env/'+k, v, log_dict['it'])
            
            # 收集 loss 指标
            for loss_key, loss_value in log_dict['loss_dict'].items():
                fabric_metrics[f'Learn/{loss_key}'] = loss_value
                if self.fabric.global_rank == 0:
                    self.writer.add_scalar(f'Learn/{loss_key}', loss_value, log_dict['it'])
            
            # 收集学习率和噪声标准差
            fabric_metrics['Learn/actor_learning_rate'] = self.actor_learning_rate
            fabric_metrics['Learn/critic_learning_rate'] = self.critic_learning_rate
            fabric_metrics['Learn/discriminator_learning_rate'] = self.discriminator_learning_rate
            fabric_metrics['Learn/mean_noise_std'] = self.actor.std.mean().item()
            
            if self.fabric.global_rank == 0:
                self.writer.add_scalar('Learn/actor_learning_rate', self.actor_learning_rate, log_dict['it'])
                self.writer.add_scalar('Learn/critic_learning_rate', self.critic_learning_rate, log_dict['it'])
                self.writer.add_scalar('Learn/discriminator_learning_rate', self.discriminator_learning_rate, log_dict['it'])
                self.writer.add_scalar('Learn/mean_noise_std', self.actor.std.mean().item(), log_dict['it'])
            
            return env_log_string

        def generate_episode_log():
            # Generate log for episode information
            ep_string = f"{'-' * width}\n"  # Add a separator line before episode info
            
            if log_dict['ep_infos']:
                # Initialize a dictionary to hold the sum and count for mean calculation
                mean_values = {key: 0.0 for key in log_dict['ep_infos'][0].keys()}
                total_episodes = 0

                for ep_info in log_dict['ep_infos']:
                    # Sum the values for mean calculation
                    for key in mean_values.keys():
                        # Check if the key is 'end_epis_length' and handle it accordingly
                        if key == 'end_epis_length':
                            # Sum the lengths of episodes
                            mean_values[key] += ep_info[key].sum().item()  # Convert tensor to scalar
                            total_episodes += ep_info[key].numel()  # Count the number of episodes
                        else:
                            mean_values[key] += (
                                        ep_info[key]  #/ ep_info['end_epis_length'] * self.env.max_episode_length 
                                                ).sum().item()  # Average for other keys

                rew_total = 0
                for key, value in mean_values.items():
                    if key.startswith('rew_'):
                        rew_total += value
                        
                mean_values['rew_total'] = rew_total
                
                # Calculate the mean for each key
                for key in mean_values.keys():
                    mean_values[key] /= total_episodes  # Mean over all episode lengths
                    # 收集到 fabric_metrics
                    if key.startswith('rew_'):
                        fabric_metrics[f'Rew/{key}'] = mean_values[key]
                    else:
                        fabric_metrics[f'Env/{key}'] = mean_values[key]
                    
                    # TensorBoard 记录 (只在主进程)
                    if self.fabric.global_rank == 0:
                        if key.startswith('rew_'):
                            self.writer.add_scalar('Rew/' + key, mean_values[key], log_dict['it'])
                        else:
                            self.writer.add_scalar('Env/' + key, mean_values[key], log_dict['it'])
                    
                    
                        
                # Prepare the string for logging
                for key, value in mean_values.items():
                    if key == 'end_epis_length': continue
                    ep_string += f"""{f'{key}:':>{pad}} {value:>10.4f} \n"""  # Print mean values with 4 decimal places
            ep_string += f"Note: reward computed per step\n"

            return ep_string

        def generate_total_time_log():
            # Calculate ETA and generate total time log
            fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
            eta = self.tot_time / (log_dict['it'] + 1) * (log_dict['num_learning_iterations'] - log_dict['it'])
            
            # 收集性能指标到 fabric_metrics
            fabric_metrics['Perf/total_fps'] = fps
            fabric_metrics['Perf/collection_time'] = log_dict['collection_time']
            fabric_metrics['Perf/learning_time'] = log_dict['learn_time']
            fabric_metrics['Perf/iter_time'] = iteration_time
            fabric_metrics['Perf/total_time'] = self.tot_time
            fabric_metrics['Perf/eta'] = eta
            
            # TensorBoard 记录 (只在主进程)
            if self.fabric.global_rank == 0:
                self.writer.add_scalar('Perf/total_fps', fps, log_dict['it'])
                self.writer.add_scalar('Perf/collection_time', log_dict['collection_time'], log_dict['it'])
                self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
                self.writer.add_scalar('Perf/iter_time', iteration_time, log_dict['it'])
                self.writer.add_scalar('Perf/total_time', self.tot_time, log_dict['it'])
        
            return (f"""{'-' * width}\n"""
                    f"""{'Total timesteps:':>{pad}} {self.tot_timesteps:.0f}\n"""  # Integer without decimal
                    f"""{'Collection time:':>{pad}} {log_dict['collection_time']:>10.4f}s\n"""  # Four decimal places
                    f"""{'Learning time:':>{pad}} {log_dict['learn_time']:>10.4f}s\n"""  # Four decimal places
                    f"""{'Iteration time:':>{pad}} {iteration_time:>10.4f}s\n"""  # Four decimal places
                    f"""{'Total time:':>{pad}} {self.tot_time:>10.4f}s\n"""  # Four decimal places
                    f"""{'ETA:':>{pad}} {eta:>10.4f}s\n"""
                    f"""{'Time Now:':>{pad}} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n""")  # Four decimal places

        # Generate all log strings
        log_string = (generate_computation_log() +
                      generate_reward_length_log() +
                      generate_env_log() +
                      generate_episode_log() +
                      generate_total_time_log() +
                      f"Logging Directory: {self.log_dir}")

        # 使用 Fabric 记录所有指标
        if fabric_metrics:
            self.fabric.log_dict(fabric_metrics, step=log_dict['it'])

        # Use rich Live to update a specific section of the console (只在主进程显示)
        if self.fabric.global_rank == 0:
            with Live(Panel(log_string, title="Training Log"), refresh_per_second=4, console=console):
                pass
    
