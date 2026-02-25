import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
from loguru import logger

from utils.devtool import *


from utils.config_utils import *  # noqa: E402, F403
@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    # import ipdb; ipdb.set_trace()
    simulator_type = config.simulator['_target_'].split('.')[-1]
    # import ipdb; ipdb.set_trace()    
        # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacGym':
        import isaacgym  # noqa: F401


    # have to import torch after isaacgym
    import torch  # noqa: E402
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge
    from utils.common import seeding
    from lightning.fabric import Fabric
    from lightning.pytorch.loggers import WandbLogger
    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if config.seed is not None:
        rank = fabric.global_rank
        if rank is None:
            rank = 0
        fabric.seed_everything(config.seed + rank)
        seeding(config.seed + rank, torch_deterministic=config.torch_deterministic)
    
    if simulator_type == 'IsaacSim':
        from isaaclab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        if fabric.world_size > 1:
            # This is needed when running with SLURM.
            # When launching multi-GPU/node jobs without SLURM, or differently, maybe this needs to be adapted accordingly.
            args_cli.distributed = True
            os.environ.LOCAL_RANK = str(fabric.local_rank)
            os.environ.RANK = str(fabric.global_rank)
        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  
        
    #     # import ipdb; ipdb.set_trace()
    # if simulator_type == 'IsaacGym':
    #     import isaacgym  # noqa: F401


    # # have to import torch after isaacgym
    # import torch  # noqa: E402
    # from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    # from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    # from humanoidverse.utils.helpers import pre_process_config
    # from humanoidverse.utils.logging import HydraLoggerBridge
        
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    if config.use_wandb:
        import wandb
        import swanlab
        project_name = f"{config.project_name}"
        run_name = f"{config.timestamp}_{config.experiment_name}_{config.log_task_name}_{config.robot.asset.robot_type}"
        wandb_dir = Path(HydraConfig.get().runtime.output_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving wandb logs to {wandb_dir}")
        swanlab.sync_wandb(wandb_run=False)
        swanlab.sync_tensorboard_torch()
        wandb.init(project=project_name, 
                # entity=config.wandb.wandb_entity,
                name=run_name,
                sync_tensorboard=True,
                config=unresolved_conf,
                dir=wandb_dir)
    
    if hasattr(config, 'device'):
        if config.device is not None:
            device = config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pre_process_config(config)

    # 如果 algo.config.teacher_ckpt 被设置，则按 eval_agent 的逻辑为 teacher 加载完整 config.yaml
    teacher_ckpt = getattr(config.algo.config, "teacher_ckpt", None)
    if teacher_ckpt is not None:
        has_teacher_config = True
        checkpoint = Path(teacher_ckpt)
        teacher_config_path = checkpoint.parent / "config.yaml"
        if not teacher_config_path.exists():
            teacher_config_path = checkpoint.parent.parent / "config.yaml"
            if not teacher_config_path.exists():
                has_teacher_config = False
                logger.error(f"[Distill] Could not find teacher config.yaml near {teacher_ckpt}")

        if has_teacher_config:
            logger.info(f"[Distill] Loading teacher training config file from {teacher_config_path}")
            with open(teacher_config_path) as f:
                teacher_config = OmegaConf.load(f)
            # 对 teacher_config 也做一次 pre_process_config，算出其 obs_dim_dict 和 obs_slices
            pre_process_config(teacher_config)
            # 挂到 algo.config 上，供 MHPPO 使用
            config.algo.config.teacher_config = teacher_config

    if config.algo.config.module_dict.actor.type == 'Transformer':
        config.env.config.use_transformer = True
    else:
        config.env.config.use_transformer = False
    
    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    # import ipdb; ipdb.set_trace()
    env: BaseEnv = instantiate(config=config.env, device=str(fabric.device)) #根据配置文件动态创建对象实例，根据配置中的 _target_ 确定要实例化的类


    experiment_save_dir = Path(config.experiment_dir)
    experiment_save_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Saving config file to {experiment_save_dir}")
    with open(experiment_save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)

    algo: BaseAlgo = instantiate(fabric=fabric, env=env, config=config.algo, log_dir=experiment_save_dir)
    algo.setup()
    algo.fabric.strategy.barrier()
    if config.checkpoint is not None:
        algo.load(config.checkpoint)
    algo.fabric.strategy.barrier()

    # handle saving config
    algo.learn()

    if simulator_type == 'IsaacSim':
        simulation_app.close()

if __name__ == "__main__":
    main()
