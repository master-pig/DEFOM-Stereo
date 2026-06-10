python evaluate_stereo.py \
  --restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
  --datasets eth3d \
  --dinov2_encoder vits \
  --scale_iters 8 \
  --idepth_scale 0.5 \
  --corr_levels 2 \
  --corr_radius 4 \
  --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
  --scale_corr_radius 2

python evaluate_stereo.py \
  --restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
  --datasets eth3d \
  --dinov2_encoder vits \
  --scale_iters 8 \
  --idepth_scale 0.5 \
  --corr_levels 2 \
  --corr_radius 4 \
  --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
  --scale_corr_radius 2 \
  --extractor_module extractor_defom \
  --save_vis_dir vis_eth3d \
  --save_vis_limit 20