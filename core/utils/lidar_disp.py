"""Load simulation LiDAR disparity PNGs and prepare DEFOM init disparity."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def read_lidar_disp_png(path):
    """Read 16-bit disparity PNG (Middlebury export convention).

    Returns:
        disp: (H, W) float32 disparity in pixels, ``disp = uint16 / 256``
        valid: (H, W) bool, True where raw value > 0
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    raw = arr.astype(np.float32)
    disp = raw / 256.0
    valid = raw > 0
    return disp, valid


def resolve_lidar_disp_path(image1_path, lidar_disp_name, lidar_root=None):
    """Resolve LiDAR disparity path from left image path."""
    if lidar_root is not None:
        scene = os.path.basename(os.path.dirname(image1_path))
        path = os.path.join(lidar_root, scene, lidar_disp_name)
    else:
        path = os.path.join(os.path.dirname(image1_path), lidar_disp_name)
    return path


def numpy_disp_to_tensor(disp, valid, device):
    disp_t = torch.from_numpy(disp).float().to(device)[None, None]
    valid_t = torch.from_numpy(valid.astype(np.float32)).to(device)[None, None]
    return disp_t, valid_t


def align_disp_to_image(disp, valid, target_hw):
    """Resize disparity / mask to match stereo image (H, W)."""
    th, tw = target_hw
    if disp.shape[-2:] == (th, tw):
        return disp, valid
    disp = F.interpolate(disp, size=(th, tw), mode='bilinear', align_corners=True)
    valid = F.interpolate(valid, size=(th, tw), mode='nearest')
    return disp, valid


def prepare_init_disp(external_disp, external_valid, danv2_io_sizes,
                      dav2_disp=None, fill_invalid_with_dav2=True, invalid_value=0.01):
    """Downsample full-resolution disparity to DAV2 output grid (oh, ow).

    Args:
        external_disp: (B, 1, H, W) pixel disparity at image resolution (padded).
        external_valid: (B, 1, H, W) float/bool mask, 1 = measured.
        danv2_io_sizes: (ih, iw, oh, ow) from ``get_danv2_io_size``.
        dav2_disp: optional (B, 1, oh, ow) DAV2 init for invalid fill.
        fill_invalid_with_dav2: use DAV2 where valid is False.
        invalid_value: constant fill when not using DAV2.

    Returns:
        (B, 1, oh, ow) tensor for iterative refinement init.
    """
    _, _, oh, ow = danv2_io_sizes
    disp_lr = F.interpolate(external_disp, size=(oh, ow), mode='bilinear', align_corners=True)
    valid_lr = F.interpolate(external_valid.float(), size=(oh, ow), mode='nearest') > 0.5

    if fill_invalid_with_dav2 and dav2_disp is not None:
        disp_lr = torch.where(valid_lr, disp_lr, dav2_disp)
    else:
        disp_lr = torch.where(valid_lr, disp_lr, torch.full_like(disp_lr, invalid_value))
    return disp_lr
