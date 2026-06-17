#!/usr/bin/env bash

# Tested config for 4 x 11GB GPUs with mixed precision.
# Validation is effectively disabled by setting val_freq > num_steps,
# so the original train_stereo.py will not enter validate_things().

CHECKPOINT_DIR=checkpoints/defomstereo_vits_middeval3_try1 && \
mkdir -p ${CHECKPOINT_DIR} && \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=9993 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 3 \
--name defomstereo_vits_middeval3_try1 \
--batch_size 4 \
--num_workers 8 \
--train_datasets middlebury_F middlebury_H middlebury_Q \
--train_folds 100 100 100 \
--num_steps 100000 \
--val_freq 200000 \
--mixed_precision \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vits \
--image_size 384 512 \
--resume_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


CHECKPOINT_DIR=checkpoints/defomstereo_vits_instereo2k_mb2014_mb2021
mkdir -p ${CHECKPOINT_DIR}

nohup env CUDA_VISIBLE_DEVICES=1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python -m torch.distributed.launch --nproc_per_node=3 --master_port=9993 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 2 \
--name defomstereo_vits_instereo2k_mb2014_mb2021 \
--batch_size 3 \
--num_workers 6 \
--train_datasets instereo2k middlebury_2014 middlebury_2021 \
--train_folds 50 200 200 \
--num_steps 100000 \
--val_freq 200000 \
--mixed_precision \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--resume_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
> ${CHECKPOINT_DIR}/train.log 2>&1 &