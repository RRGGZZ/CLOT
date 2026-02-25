from __future__ import annotations
from copy import deepcopy

from typing import List
import torch
import torch.nn as nn
from torch.distributions import Normal

from .modules import BaseModule, Transformer

class PPOActor(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions,
                init_noise_std,
                obs_slices,
                history_num=None,
                future_num=None):
        super(PPOActor, self).__init__()
        # import ipdb; ipdb.set_trace()
        module_config_dict = self._process_module_config(module_config_dict, num_actions)
        if module_config_dict['type'] == 'MLP':
            self.net_type = 'MLP'
            self.actor_module = BaseModule(obs_dim_dict, module_config_dict)
        elif module_config_dict['type'] == 'Transformer':
            self.net_type = 'Transformer'
            self.actor_module = Transformer(history_num, future_num, module_config_dict, obs_slices)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _check_tensor_anomalies(self, tensor, name="Tensor"):
        """检查tensor中的异常值"""
        if tensor is None:
            print(f"   {name}: None")
            return
        
        print(f"   {name}: shape={tensor.shape}, range=[{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
        print(f"     均值: {tensor.mean().item():.6f}, 标准差: {tensor.std().item():.6f}")
        
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        
        if nan_count > 0 or inf_count > 0:
            print(f"     🚨 包含NaN: {nan_count}, Inf: {inf_count}")

    def update_distribution(self, actor_obs):
        # import ipdb; ipdb.set_trace()
        # print(actor_obs['current'].shape)
        # print(actor_obs['past'].shape)
        # print(actor_obs['future'].shape)
        # print(self.net_type)
        if self.net_type == 'MLP':
            mean = self.actor(actor_obs)
        elif self.net_type == 'Transformer':
            mean = self.actor(actor_obs['actor_obs_current'], actor_obs['actor_obs_past'], actor_obs['actor_obs_future'])

        # 数据类型检查（可选）
        # if mean.dtype != torch.float32:
        #     mean = mean.to(dtype=torch.float32)

        # 完整的NaN和Inf检查
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            # 详细记录异常信息
            nan_count = torch.isnan(mean).sum().item()
            inf_count = torch.isinf(mean).sum().item()
            pos_inf_count = torch.isposinf(mean).sum().item()
            neg_inf_count = torch.isneginf(mean).sum().item()
            
            print(f"🚨 Actor输出包含异常值!")
            print(f"   网络类型: {self.net_type}")
            print(f"   形状: {mean.shape}")
            print(f"   NaN数量: {nan_count}")
            print(f"   Inf数量: {inf_count} (正无穷: {pos_inf_count}, 负无穷: {neg_inf_count})")
            print(f"   数值范围: [{mean.min().item():.6f}, {mean.max().item():.6f}]")
            
            # 检查输入数据
            if self.net_type == 'MLP':
                self._check_tensor_anomalies(actor_obs, "Actor输入(actor_obs)")
            else:
                self._check_tensor_anomalies(actor_obs['actor_obs_current'], "Actor当前输入")
                self._check_tensor_anomalies(actor_obs['actor_obs_past'], "Actor过去输入")
                self._check_tensor_anomalies(actor_obs['actor_obs_future'], "Actor未来输入")
            
            # 修复异常值，防止训练崩溃
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
            mean = torch.clamp(mean, -10.0, 10.0)  # 限制在合理范围内
            
            print("✅ 已修复异常值，继续训练...")

        # 额外的安全检查（即使没有NaN/Inf也检查数值范围）
        elif torch.abs(mean).max() > 1000:  # 数值过大警告
            max_val = torch.abs(mean).max().item()
            print(f"⚠️ Actor输出数值过大: {max_val:.6f}")
            mean = torch.clamp(mean, -100.0, 100.0)  # 限制极端值

        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        return actions_mean

    def act_inference_transformer(self, actor_obs_current, actor_obs_past, actor_obs_future):
        actions_mean = self.actor(actor_obs_current, actor_obs_past, actor_obs_future)
        return actions_mean
    
    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')

class PPOCritic(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                obs_slices,
                history_num=None,
                future_num=None):
        super(PPOCritic, self).__init__()

        if module_config_dict['type'] == 'MLP':
            self.net_type = 'MLP'
            self.critic_module = BaseModule(obs_dim_dict, module_config_dict)
        elif module_config_dict['type'] == 'Transformer':
            self.net_type = 'Transformer'
            self.critic_module = Transformer(history_num, future_num, module_config_dict, obs_slices)

    @property
    def critic(self):
        return self.critic_module
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, critic_obs, **kwargs):
        if self.net_type == 'MLP':
            value = self.critic(critic_obs)
        elif self.net_type == 'Transformer':
            value = self.critic(critic_obs['critic_obs_current'], critic_obs['critic_obs_past'], critic_obs['critic_obs_future'])
        return value

class GPOPolicy(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions,
                init_noise_std):
        super(GPOPolicy, self).__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)

        obs_dim_dict_ = self.generate_full_obs_dict(obs_dim_dict)

        self.policy_module = BaseModule(obs_dim_dict_, module_config_dict)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict
    
    def generate_full_obs_dict(self,obs_dim_dict):
        actor_dim = obs_dim_dict.get("actor_obs", 0)
        critic_dim = obs_dim_dict.get("critic_obs", 0)
        obs_dim_dict["full_obs"] = actor_dim + critic_dim + 1
        return obs_dim_dict
    
    @property
    def policy(self):
        return self.policy_module
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs, is_guider=False):
        # input obs is a dict with keys "actor_obs", "critic_obs" or "full_obs"
        if is_guider:
            indicator = torch.ones(obs["actor_obs"].shape[:-1] + (1,), device=obs["actor_obs"].device)
            obs_input = torch.cat([obs["actor_obs"], obs["critic_obs"], indicator], dim=-1)
        else:
            indicator = torch.zeros(obs["actor_obs"].shape[:-1] + (1,), device=obs["actor_obs"].device)
            obs_input = torch.cat([obs["actor_obs"], torch.zeros_like(obs["critic_obs"]), indicator], dim=-1)

        mean = self.policy(obs_input)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs, is_guider=False,**kwargs):
        self.update_distribution(obs, is_guider)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        # use learner obs
        indicator = torch.zeros(obs["actor_obs"].shape[:-1] + (1,), device=obs["actor_obs"].device)
        obs_input = torch.cat([obs["actor_obs"], torch.zeros_like(obs["critic_obs"]), indicator], dim=-1)
        actions_mean = self.policy(obs_input)
        return actions_mean
    
    def to_cpu(self):
        self.policy = deepcopy(self.policy).to('cpu')
        self.std.to('cpu')

class GPOValue(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict):
        super(GPOValue, self).__init__()
        
        obs_dim_dict_ = self.generate_full_obs_dict(obs_dim_dict)

        self.value_module = BaseModule(obs_dim_dict_, module_config_dict)

    def generate_full_obs_dict(self,obs_dim_dict):
        actor_dim = obs_dim_dict.get("actor_obs", 0)
        critic_dim = obs_dim_dict.get("critic_obs", 0)
        obs_dim_dict["full_obs"] = actor_dim + critic_dim + 1
        return obs_dim_dict
    
    @property
    def value(self):
        return self.value_module
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, obs, **kwargs):
        indicator = torch.ones(obs["actor_obs"].shape[:-1] + (1,), device=obs["actor_obs"].device)
        obs_input = torch.cat([obs["actor_obs"], obs["critic_obs"], indicator], dim=-1)
        value = self.value(obs_input)
        return value


class PPOActorFixSigma(PPOActor):
    def __init__(self,                 
                 obs_dim_dict,
                network_dict,
                network_load_dict,
                num_actions,):
        super(PPOActorFixSigma, self).__init__(obs_dim_dict, network_dict, network_load_dict, num_actions, 0.0)
        
    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict)['head']
        self.distribution = mean

    @property
    def action_mean(self):
        return self.distribution
    
    def get_actions_log_prob(self, actions):
        raise NotImplementedError
    
    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return self.distribution

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim
            max_level
        """
        super(SinusoidalEmbedding, self).__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_level = (embed_dim // 2)
        self.register_buffer("freqs", 2 ** torch.arange(self.max_level).float())

    def forward(self, phase):
        """
        Args:
            phase: (...,) tensor in [0, 1]
        Returns:
            embedding: (..., embed_dim)
        """
        angles = 2 * torch.pi * phase * self.freqs  # (..., max_level)
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        return torch.cat([sin_enc, cos_enc], dim=-1)  # (..., 2 * max_level)

class SinusoidalEmbeddingV2(nn.Module):
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim
            max_level
        """
        super(SinusoidalEmbeddingV2, self).__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_level = (embed_dim // 2)
        self.register_buffer("freqs", 1/ (10000 ** (torch.arange(self.max_level).float()/self.max_level)))

    def forward(self, phase):
        """
        Args:
            phase: (...,) tensor in [0, 1]
        Returns:
            embedding: (..., embed_dim)
        """
        angles = 2 * torch.pi * phase * self.freqs  # (..., max_level)
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        return torch.cat([sin_enc, cos_enc], dim=-1)  # (..., 2 * max_level)

class Identity(nn.Module):
    def forward(self, x):
        return x

class ModuleWithPhaseEmbedding(nn.Module):
    def __init__(self, module, embed_type, embed_dim):
        super(ModuleWithPhaseEmbedding, self).__init__()
        self.embed_type = embed_type
        self.embed_dim = embed_dim
        self.module = module  # BaseModule

        if self.embed_type == 'Original':
            self.phase_embedder = Identity()
            assert self.embed_dim == 1, "embed_dim should be 1 for Original embedding"
        elif self.embed_type == 'Sinusoidal':
            self.phase_embedder = SinusoidalEmbedding(self.embed_dim)  # Standard SinusoidalEmbedding 

        elif self.embed_type == 'Learnable':
            self.phase_embedder = nn.Sequential(nn.Linear(1, self.embed_dim*2),
                                                nn.ReLU(),
                                                nn.Linear(self.embed_dim*2, self.embed_dim)) # Learnable PhaseEmbedding
            # self.phase_embedder = nn.Linear(1, self.embed_dim)  # Learnable PhaseEmbedding
        # stay tuned
        else:
            raise ValueError(f"Unknown embed_type {self.embed_type}")
        
    def forward(self, obs):

        phase = obs[..., -1:]  # assume phase is the last element in the observation
        phase_embedding = self.phase_embedder(phase)
        # import ipdb; ipdb.set_trace()
        obs_input = obs[..., :-1] 
        obs_input = torch.cat([obs_input, phase_embedding], dim=-1) 
        
        return self.module(obs_input)


class PhaseAwareActor(nn.Module):
    def __init__(self,
                 obs_dim_dict,
                 module_config_dict,
                 num_actions,
                 init_noise_std,
                 embed_type='Sinusoidal',
                 phase_embed_dim=16): 
        super(PhaseAwareActor, self).__init__()

        module_config_dict = self._process_module_config(module_config_dict, num_actions)
       
        # original actor_module
        self.actor_module = BaseModule(obs_dim_dict, module_config_dict)
        
        #  ActorWithPhaseEmbedding = actor + phase_embedder
        self.actor_with_phase = ModuleWithPhaseEmbedding(self.actor_module, embed_type, phase_embed_dim)
                                                        

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_with_phase
    
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self, actor_obs):
        return self.actor(actor_obs)
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        return actions_mean
    
    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')


class PhaseAwareCritic(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                embed_type = 'Sinusoidal',
                phase_embed_dim=16):
        super(PhaseAwareCritic, self).__init__()

        self.critic_module = BaseModule(obs_dim_dict, module_config_dict)

        self.critic_with_phase = ModuleWithPhaseEmbedding(self.critic_module,embed_type, phase_embed_dim)

    @property
    def critic(self):
        return self.critic_with_phase
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, critic_obs, **kwargs):
        value = self.critic(critic_obs)
        return value



class PhaseEmbeddingModuleV2(nn.Module):
    def __init__(self, phase_pos:List[int], embed_type: str, embed_dim: int):
        super(PhaseEmbeddingModuleV2, self).__init__()
        self.phase_pos = phase_pos
        self.embed_type = embed_type
        self.embed_dim = embed_dim

        if self.embed_type == 'Original':
            raise ValueError("Original embedding is not valid for PhaseEmbeddingModuleV2")
        elif self.embed_type == 'Sinusoidal':
            self.phase_embedder = SinusoidalEmbedding(self.embed_dim)  # Standard SinusoidalEmbedding 
        elif self.embed_type == 'SinusoidalV2':
            self.phase_embedder = SinusoidalEmbeddingV2(self.embed_dim)  # Standard SinusoidalEmbedding 

        elif self.embed_type == 'Learnable':
            self.phase_embedder = nn.Sequential(nn.Linear(1, self.embed_dim*2),
                                                nn.ReLU(),
                                                nn.Linear(self.embed_dim*2, self.embed_dim)) # Learnable PhaseEmbedding
        else:
            raise ValueError(f"Unknown embed_type {self.embed_type}")
        
    def forward(self, obs):
        # Extract phases at specified positions using tensor indexing
        phases = obs[..., self.phase_pos].unsqueeze(-1)  # Shape: [..., num_phases, 1]
        
        # Embed all phases at once
        phase_embeddings = self.phase_embedder(phases)  # Shape: [..., num_phases, embed_dim]
        
        # Reshape embeddings to combine all phase embeddings
        combined_embedding = phase_embeddings.reshape(*phases.shape[:-2], -1)  # Shape: [..., num_phases * embed_dim]
        
        
        # Concatenate observation with phase embeddings
        obs_with_pe = torch.cat([obs, combined_embedding], dim=-1)
        print(f"DEBUG: {self.phase_pos=}  \t| {phases[:5].squeeze()=} \t| {obs_with_pe.shape=}")
        return obs_with_pe

class PhaseAwareActorV2(PPOActor):
    def __init__(self,
                 obs_dim_dict,
                 module_config_dict,
                 num_actions,
                 init_noise_std,
                 actor_phase_pos:int,
                 phase_embed_type:str,
                 phase_embed_dim:int):
        obs_dim_with_pe = obs_dim_dict['actor_obs'] + phase_embed_dim
        super(PhaseAwareActorV2, self).__init__({
            'actor_obs': obs_dim_with_pe
        }, module_config_dict, num_actions, init_noise_std)
        
        # Create phase embedding module
        self.phase_embedder = PhaseEmbeddingModuleV2([actor_phase_pos], phase_embed_type, phase_embed_dim)
        
        # Create actor module with phase embedding
        self.actor_with_phase = nn.Sequential(self.phase_embedder, self.actor_module)

    @property
    def actor(self):
        return self.actor_with_phase

class PhaseAwareCriticV2(PPOCritic):
    def __init__(self,
                 obs_dim_dict,
                 module_config_dict,
                 critic_phase_pos:int,
                 phase_embed_type:str,
                 phase_embed_dim:int):
        obs_dim_with_pe = obs_dim_dict['critic_obs'] + phase_embed_dim
        super(PhaseAwareCriticV2, self).__init__({
            'critic_obs': obs_dim_with_pe
        }, module_config_dict)
        
        # Create phase embedding module
        self.phase_embedder = PhaseEmbeddingModuleV2([critic_phase_pos], phase_embed_type, phase_embed_dim)
        
        # Create critic module with phase embedding
        self.critic_with_phase = nn.Sequential(self.phase_embedder, self.critic_module)

    @property
    def critic(self):
        return self.critic_with_phase