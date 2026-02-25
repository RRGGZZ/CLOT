HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=8 humanoidverse/train_agent.py \
+simulator=mjlab \
+exp=motion_tracking_amp_transformer \
+terrain=terrain_locomotion_plane \
project_name=G1-amp-transformer \
num_envs=4096 \
+obs=motion_tracking/main_g1_amp \
+robot=g1/g1_23dof_lock_wrist \
+domain_rand=main_g1 \
+rewards=motion_tracking/main_g1 \
experiment_name=train-g1 \
robot.motion.motion_file="human_motion/g1_data_50fps_cont_mask.pkl" \
seed=1 
# +device=cuda:0
    