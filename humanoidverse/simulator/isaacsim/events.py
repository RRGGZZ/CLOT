from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Literal
from loguru import logger
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedEnv

def resolve_dist_fn(
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    dist_fn = math_utils.sample_uniform

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")

    return dist_fn
def randomize_rigid_body_mass(env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    num_envs: int = 1,
    ):
    # sim: IsaacSim 实例
    # asset_cfg: SceneEntityCfg，指定 body_names
    # mass_distribution_params: (min, max)
    # operation: "scale"
    if env_ids is None:
        env_ids = torch.arange(num_envs, device='cpu')
    else:
        env_ids = env_ids.to('cpu')
    asset = env.scene[asset_cfg.name]

    masses = asset.root_physx_view.get_masses().to('cpu')  # 确保在正确的设备上
    if max(env_ids) < 10:
        logger.debug(f"Before randomize masses: {masses[env_ids]}")
    masses_new = masses.clone()
    scales = torch.zeros(len(env_ids), len(env.domain_rand_config.randomize_link_body_names), device='cpu')
    for i, body_name in enumerate(env.domain_rand_config.randomize_link_body_names):
        body_index = env._body_list.index(body_name)
        assert body_index != -1
        
        # 为每个环境生成不同的 scale（在正确的设备上）
        scales[env_ids[:, None], i] = torch.rand([len(env_ids),1], device='cpu') * (mass_distribution_params[1] - mass_distribution_params[0]) + mass_distribution_params[0]
        
        # 实际应用到物理属性（每个环境使用各自的 scale）
        masses_new[env_ids[:, None], body_index] = masses[env_ids[:, None], body_index] * scales[env_ids[:, None], i]
    # 记录scale
    env._link_mass_scale = scales.to(env.sim_device)
        
    # PhysX API 需要 CPU tensor
    # masses_new_cpu = masses_new.cpu()
    # env_ids_cpu = env_ids.cpu()
    asset.root_physx_view.set_masses(masses_new, env_ids)
    if max(env_ids) < 10:
        logger.debug(f"Before randomize masses: {masses[:3,:3]}")
        logger.debug(f"After randomize masses: {masses_new[:3,:3]}")
        logger.debug(f"Get masses scale: {env._link_mass_scale[:3,:3]}")
def randomize_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    num_envs: int = 1, # number of environments
):
    """Randomize the com of the bodies by adding, scaling or setting random values.

    This function allows randomizing the center of mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms()
    # apply randomization on default values
    # import ipdb; ipdb.set_trace()
    coms[env_ids[:, None], body_ids] = env.default_coms[env_ids[:, None], body_ids].clone()
    if max(env_ids) < 10:
        logger.debug(f"Before randomize body coms: {coms[:3,:3]}")
    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (distribution_params[0].to(coms.device), distribution_params[1].to(coms.device))
    base_com_bias = torch.zeros((num_envs, 3), device=coms.device, dtype=coms.dtype,requires_grad=False)
    base_com_bias[env_ids, :] = dist_fn(
        *distribution_params, (env_ids.shape[0], base_com_bias.shape[1]), device=coms.device
    )

    # sample from the given range
    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += base_com_bias[env_ids[:, None], :]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = base_com_bias[env_ids[:, None], :]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= base_com_bias[env_ids[:, None], :]
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env._base_com_bias = base_com_bias.to(env.sim_device)
    asset.root_physx_view.set_coms(coms, env_ids)
    if max(env_ids) < 10:
        logger.debug(f"Base com bias: {env._base_com_bias[:3,:3]}")
        logger.debug(f"After randomize body coms: {coms[:3,:3]}")


def randomize_joint_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float],
    operation: Literal["add", "abs", "scale"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    num_envs: int = 1,
):
    """自定义关节摩擦随机化函数，将friction值保存到env.simulator.friction_coeffs中
    
    参考 IsaacGym 的实现方式，使用 bucket 方法来减少不同的 friction 值数量
    """
    # 获取 asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 解析环境 IDs (PhysX API 需要 CPU tensor)
    if env_ids is None:
        env_ids = torch.arange(num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    
    # 只在第一次调用时初始化 friction_coeffs（startup 模式）
    if not hasattr(env.simulator, '_friction_initialized'):
        logger.info(f"初始化 friction 随机化，范围: {friction_distribution_params}")
        
        # 参考 IsaacGym 的实现：使用 bucket 方法
        friction_range = friction_distribution_params
        num_buckets = 64
        bucket_ids = torch.randint(0, num_buckets, (num_envs, 1), device="cpu")
        
        # 生成 friction buckets (在 CPU 上)
        friction_buckets = torch.rand(num_buckets, 1, device="cpu") * (friction_range[1] - friction_range[0]) + friction_range[0]
        # 保存到 GPU (供观测使用)
        env.simulator.friction_coeffs = friction_buckets[bucket_ids].to(env.sim_device)
        
        env.simulator._friction_initialized = True
        logger.info(f"Friction coeffs shape: {env.simulator.friction_coeffs.shape}")
        logger.info(f"Friction coeffs 范例 (前3个环境): {env.simulator.friction_coeffs[:3].flatten()}")
    
    # 获取当前的 DOF 摩擦参数 (使用正确的 API 方法名)
    joint_friction = asset.root_physx_view.get_dof_friction_coefficients()  # 返回 CPU tensor
    
    # 为每个环境设置 friction
    for env_id in env_ids:
        friction_value = env.simulator.friction_coeffs[env_id].item()
        
        # 应用到所有关节（根据 asset_cfg.joint_ids）
        if asset_cfg.joint_ids == slice(None):
            joint_indices = torch.arange(asset.num_joints, dtype=torch.int, device="cpu")
        else:
            joint_indices = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device="cpu")
        
        if operation == "scale":
            joint_friction[env_id, joint_indices] *= friction_value
        elif operation == "add":
            joint_friction[env_id, joint_indices] += friction_value
        elif operation == "abs":
            joint_friction[env_id, joint_indices] = friction_value
    
    # 设置回物理仿真 (PhysX API 需要 CPU tensor)
    asset.root_physx_view.set_dof_friction_coefficients(joint_friction, env_ids)
    
    if max(env_ids) < 3:
        logger.debug(f"设置关节 friction 完成，环境 {env_ids[:3]} 的 friction 系数: {env.simulator.friction_coeffs[env_ids[:3]].flatten()}")

def randomize_rigid_body_restitution(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    restitution_distribution_params: tuple[float, float],
    operation: Literal["add", "abs", "scale"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    num_envs: int = 1,
):
    """自定义刚体恢复系数随机化函数，将restitution值保存到env.simulator.restitution_coeffs中
    
    参考 IsaacGym 的实现方式，为每个环境随机生成 restitution 值
    """
    # 获取 asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 解析环境 IDs (PhysX API 需要 CPU tensor)
    if env_ids is None:
        env_ids = torch.arange(num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    
    # 只在第一次调用时初始化 restitution_coeffs（startup 模式）
    if not hasattr(env.simulator, '_restitution_initialized'):
        logger.info(f"初始化 restitution 随机化，范围: {restitution_distribution_params}")
        
        # 参考 IsaacGym 的实现：为每个环境直接随机生成值（不使用 bucket）
        restitution_range = restitution_distribution_params
        restitution_values = torch.rand(num_envs, 1, device="cpu") * (restitution_range[1] - restitution_range[0]) + restitution_range[0]
        
        # 保存为 (num_envs, 1, 1) 格式以匹配 IsaacGym (保存到 GPU 供观测使用)
        env.simulator.restitution_coeffs = restitution_values.unsqueeze(-1).to(env.sim_device)
        
        env.simulator._restitution_initialized = True
        logger.info(f"Restitution coeffs shape: {env.simulator.restitution_coeffs.shape}")
        logger.info(f"Restitution coeffs 范例 (前3个环境): {env.simulator.restitution_coeffs[:3].flatten()}")
        logger.info(f"Restitution range: [{env.simulator.restitution_coeffs.min().item():.4f}, {env.simulator.restitution_coeffs.max().item():.4f}]")
    
    # 获取当前的刚体材质属性 (返回 CPU tensor)
    restitutions = asset.root_physx_view.get_restitutions()
    
    # 解析 body indices (在 CPU 上)
    if asset_cfg.body_ids == slice(None):
        body_indices = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_indices = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    
    # 为每个环境设置 restitution
    for env_id in env_ids:
        restitution_value = env.simulator.restitution_coeffs[env_id].item()
        
        if operation == "abs":
            restitutions[env_id, body_indices] = restitution_value
        elif operation == "scale":
            restitutions[env_id, body_indices] *= restitution_value
        elif operation == "add":
            restitutions[env_id, body_indices] += restitution_value
    
    # 设置回物理仿真 (PhysX API 需要 CPU tensor)
    asset.root_physx_view.set_restitutions(restitutions, env_ids)
    
    if max(env_ids) < 3:
        logger.debug(f"设置刚体 restitution 完成，环境 {env_ids[:3]} 的 restitution 系数: {env.simulator.restitution_coeffs[env_ids[:3]].flatten()}")
