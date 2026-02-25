
HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=8 humanoidverse/train_agent.py \
+simulator=mjlab \
+exp=motion_tracking_amp_transformer \
+terrain=terrain_locomotion_plane \
project_name=mjlab-amp-transformer \
num_envs=4096 \
+obs=motion_tracking/main_adam_amp \
+robot=adam/adam_pro \
+domain_rand=main_adam \
+rewards=motion_tracking/main_adam \
experiment_name=train-adam-amp-transformer \
robot.motion.motion_file="human_motion/adam_data_30fps_cont_mask.pkl" \
seed=1 
# +checkpoint="logs/mjlab-mlp-without-amp/20260109_021122-train-motion-1206-motion_tracking-adam_pro/model_41000.pt"