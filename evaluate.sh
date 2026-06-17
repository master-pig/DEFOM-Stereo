CUDA_VISIBLE_DEVICES=2 \
python evaluate_stereo.py \
  --restore_ckpt checkpoints/defomstereo_vitl_middlebury.pth \
  --datasets eth3d \
  --dinov2_encoder vitl \
  --scale_iters 8 \
  --idepth_scale 0.5 \
  --corr_levels 2 \
  --corr_radius 4 \
  --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
  --scale_corr_radius 2\
  --extractor_module extractor_defom

CUDA_VISIBLE_DEVICES=2 \
python evaluate_stereo.py \
--restore_ckpt checkpoints/defomstereo_vits_instereo2k_mb2014_mb2021/checkpoint_latest.pth  \
--datasets eth3d \
--dinov2_encoder vits \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2




# CUDA_VISIBLE_DEVICES=2 \
# python evaluate_stereo.py \
#   --restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
#   --datasets eth3d \
#   --dinov2_encoder vits \
#   --scale_iters 8 \
#   --idepth_scale 0.5 \
#   --corr_levels 2 \
#   --corr_radius 4 \
#   --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
#   --scale_corr_radius 2\
#   --extractor_module extractor_defom

# CUDA_VISIBLE_DEVICES=2 \
# python evaluate_stereo.py \
#   --restore_ckpt checkpoints/defomstereo_vits_middeval3_try1/defomstereo_vits_middeval3_try1_20000.pth \
#   --datasets eth3d \
#   --dinov2_encoder vits \
#   --scale_iters 8 \
#   --idepth_scale 0.5 \
#   --corr_levels 2 \
#   --corr_radius 4 \
#   --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
#   --scale_corr_radius 2


# CUDA_VISIBLE_DEVICES=2 \
# python evaluate_stereo.py \
#   --restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
#   --datasets middlebury_Q \
#   --dinov2_encoder vits \
#   --scale_iters 8 \
#   --idepth_scale 0.5 \
#   --corr_levels 2 \
#   --corr_radius 4 \
#   --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
#   --scale_corr_radius 2 \
#   --extractor_module extractor_defom \


# CUDA_VISIBLE_DEVICES=2 \
# python evaluate_stereo.py \
#   --restore_ckpt checkpoints/defomstereo_vits_middeval3_try1/checkpoint_latest.pth \
#   --datasets middlebury_Q \
#   --dinov2_encoder vits \
#   --scale_iters 8 \
#   --idepth_scale 0.5 \
#   --corr_levels 2 \
#   --corr_radius 4 \
#   --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
#   --scale_corr_radius 2 \

# CUDA_VISIBLE_DEVICES=2 \
# python evaluate_stereo.py \
#   --restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
#   --datasets middlebury_H \
#   --dinov2_encoder vits \
#   --scale_iters 8 \
#   --idepth_scale 0.5 \
#   --corr_levels 2 \
#   --corr_radius 4 \
#   --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
#   --scale_corr_radius 2 \
#   --extractor_module extractor_defom \


# CUDA_VISIBLE_DEVICES=2 \
# python evaluate_stereo.py \
#   --restore_ckpt checkpoints/defomstereo_vits_middeval3_try1/checkpoint_latest.pth \
#   --datasets middlebury_H \
#   --dinov2_encoder vits \
#   --scale_iters 8 \
#   --idepth_scale 0.5 \
#   --corr_levels 2 \
#   --corr_radius 4 \
#   --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
#   --scale_corr_radius 2 \

# CUDA_VISIBLE_DEVICES=2 \
# python evaluate_stereo.py \
#   --restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
#   --datasets middlebury_F \
#   --dinov2_encoder vits \
#   --scale_iters 8 \
#   --idepth_scale 0.5 \
#   --corr_levels 2 \
#   --corr_radius 4 \
#   --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
#   --scale_corr_radius 2 \
#   --extractor_module extractor_defom \


# CUDA_VISIBLE_DEVICES=2 \
# python evaluate_stereo.py \
#   --restore_ckpt checkpoints/defomstereo_vits_middeval3_try1/checkpoint_latest.pth \
#   --datasets middlebury_F \
#   --dinov2_encoder vits \
#   --scale_iters 8 \
#   --idepth_scale 0.5 \
#   --corr_levels 2 \
#   --corr_radius 4 \
#   --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
#   --scale_corr_radius 2 \

python evaluate_stereo.py \
  --restore_ckpt checkpoints/defomstereo_vits_middeval3_try1/defomstereo_vits_middeval3_try1_50000.pth \
  --datasets eth3d \
  --dinov2_encoder vits \
  --scale_iters 8 \
  --idepth_scale 0.5 \
  --corr_levels 2 \
  --corr_radius 4 \
  --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
  --scale_corr_radius 2

python evaluate_stereo.py \
  --restore_ckpt checkpoints/defomstereo_vits_middeval3_try1/defomstereo_vits_middeval3_try1_60000.pth \
  --datasets eth3d \
  --dinov2_encoder vits \
  --scale_iters 8 \
  --idepth_scale 0.5 \
  --corr_levels 2 \
  --corr_radius 4 \
  --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
  --scale_corr_radius 2 

python evaluate_stereo.py \
  --restore_ckpt checkpoints/defomstereo_vits_middeval3_try1/defomstereo_vits_middeval3_try1_70000.pth \
  --datasets eth3d \
  --dinov2_encoder vits \
  --scale_iters 8 \
  --idepth_scale 0.5 \
  --corr_levels 2 \
  --corr_radius 4 \
  --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
  --scale_corr_radius 2 