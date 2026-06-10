"""
Evaluate DEFOM-Stereo on Middlebury with dense LiDAR disparity initialization.

This script keeps the pretrained DEFOM-Stereo weights unchanged and only
changes how the initial disparity is provided to the model:

1. The normal DAV2 initialization can still be used for comparison.
2. The LiDAR path injects a dense disparity PNG before iterative refinement.
3. Metrics are reported scene-by-scene and then averaged across the split.

Typical usage:
    python evaluate_stereo_dense_lidar.py \\
        --restore_ckpt checkpoints/defomstereo_vitl_middlebury.pth \\
        --middlebury_root ./datasets/Middlebury \\
        --split 2014 \\
        --lidar_disp_name disp0_lidar_dense.png

Flat scene folders (for example ``dataset/Adirondack-perfect/im0.png``):
    python evaluate_stereo_dense_lidar.py \\
        --restore_ckpt checkpoints/defomstereo_vitl_middlebury.pth \\
        --middlebury_root ./dataset \\
        --split custom \\
        --gt_disp_name disp0_gt.png
"""

from __future__ import print_function, division

import argparse
from glob import glob
import logging
import os
import os.path as osp
import time

import numpy as np
import torch

from tqdm import tqdm

from core.defom_stereo import DEFOMStereo, autocast
from core.utils.utils import InputPadder
from core.utils import frame_utils
from core.utils.lidar_disp import (
    read_lidar_disp_png,
    resolve_lidar_disp_path,
    numpy_disp_to_tensor,
    align_disp_to_image,
)
import core.stereo_datasets as datasets
from evaluate_stereo import count_parameters


def _read_gt_disp(path):
    """Read ground-truth disparity in the formats used by this evaluation.

    The standard Middlebury loader handles PFM files. For custom exports, we
    also support 16-bit PNG files that follow the same disparity convention as
    the LiDAR input helpers (stored value / 256.0).

    This keeps the evaluation script compatible with both:
    - official Middlebury-style PFM ground truth
    - converted custom datasets where GT has already been exported to PNG
    """
    if path.lower().endswith(".png"):
        return read_lidar_disp_png(path)
    disp, valid = frame_utils.readDispMiddlebury(path)
    return np.asarray(disp, dtype=np.float32), np.asarray(valid, dtype=bool)


class MiddleburyDenseLidarEval(datasets.StereoDataset):
    """Dataset adapter for flat custom scene folders.

    The built-in ``datasets.Middlebury`` class already covers official
    Middlebury layouts such as ``2014`` and ``2021``. This small adapter is
    only needed when scenes are stored directly as:

        root/<scene>/im0.png
        root/<scene>/im1.png
        root/<scene>/disp0GT.pfm or disp0_gt.png

    It produces the same sample structure as the rest of the DEFOM dataset
    code, so the evaluation loop can stay unchanged.

    Expected folder shape:

        root/<scene>/im0.png
        root/<scene>/im1.png
        root/<scene>/disp0GT.pfm or disp0_gt.png or disp0.pfm

    The class only discovers files and stores paths. The actual image and
    disparity loading is still handled by the parent ``StereoDataset`` logic.
    """

    def __init__(
        self,
        root,
        gt_disp_name="disp0GT.pfm",
        fallback_gt=("disp0_gt.png", "disp0.pfm"),
    ):
        super(MiddleburyDenseLidarEval, self).__init__(
            aug_params=None, sparse=True, reader=_read_gt_disp, is_eval=True
        )
        im0_list = sorted(glob(osp.join(root, "*", "im0.png")))
        assert len(im0_list) > 0, f"No im0.png under {root}/*/"
        for im0 in im0_list:
            scene_dir = osp.dirname(im0)
            im1 = osp.join(scene_dir, "im1.png")
            assert osp.isfile(im1), im1

            # Try the user-specified GT filename first, then fall back to a few
            # common names used by converted Middlebury-style datasets.
            disp_path = osp.join(scene_dir, gt_disp_name)
            if not osp.isfile(disp_path):
                for name in fallback_gt:
                    cand = osp.join(scene_dir, name)
                    if osp.isfile(cand):
                        disp_path = cand
                        break
            assert osp.isfile(disp_path), f"No GT disparity in {scene_dir}"
            self.image_list.append([im0, im1])
            self.disparity_list.append([disp_path])


def build_eval_dataset(args):
    """Create the evaluation dataset chosen by ``--split``.

    ``custom`` means a flat scene folder layout and uses the adapter above.
    All official Middlebury splits reuse the repository's standard dataset
    loader so behavior stays aligned with the original evaluation script.

    Returns:
        A dataset instance that yields samples with the same keys expected by
        ``validate_middlebury_lidar``.
    """
    if args.split == "custom":
        return MiddleburyDenseLidarEval(
            args.middlebury_root,
            gt_disp_name=args.gt_disp_name,
            fallback_gt=("disp0_gt.png", "disp0.pfm", "disp0GT.pfm"),
        )
    return datasets.Middlebury(
        aug_params=None,
        root=args.middlebury_root,
        split=args.split,
        is_eval=True,
    )


def load_lidar_batch(image1, image1_path, args, device):
    """Load LiDAR disparity for one scene and align it to the left image.

    Args:
        image1: Left image tensor with shape ``(1, 3, H, W)``.
        image1_path: Source path of the left image. This is used to derive the
            LiDAR file path when ``--lidar_root`` is not provided.
        args: Parsed command line arguments.
        device: CUDA device used by the current sample.

    Returns:
        ``(disp, valid, lidar_path)`` where ``disp`` and ``valid`` are tensors
        of shape ``(1, 1, H, W)``. ``disp`` stores disparity in pixel units
        and ``valid`` marks pixels that have a usable LiDAR measurement.
    """
    lidar_path = resolve_lidar_disp_path(
        image1_path, args.lidar_disp_name, lidar_root=args.lidar_root
    )
    if not osp.isfile(lidar_path):
        raise FileNotFoundError(f"LiDAR disparity not found: {lidar_path}")

    # Convert the PNG to tensors and make sure its spatial size matches the
    # image used by the model before padding is applied. The model expects the
    # initialization to be aligned with the left image, not just with the raw
    # LiDAR file resolution.
    disp_np, valid_np = read_lidar_disp_png(lidar_path)
    disp, valid = numpy_disp_to_tensor(disp_np, valid_np, device)
    disp, valid = align_disp_to_image(disp, valid, image1.shape[-2:])
    return disp, valid, lidar_path


@torch.no_grad()
def run_inference(
    model,
    image1,
    image2,
    padder,
    external_disp,
    external_valid,
    iters,
    scale_iters,
    mixed_prec,
    use_lidar,
    lidar_fill_dav2,
):
    """Run one forward pass and return the final disparity prediction.

    This helper centralizes the part of the workflow that must stay consistent
    between LiDAR-init and DAV2-init evaluation:

    1. Pad inputs so shapes are compatible with the network.
    2. Optionally attach external LiDAR disparity and validity masks.
    3. Run the model in ``test_mode`` so only the last disparity is returned.
    4. Remove padding before metric computation.

    The LiDAR tensors must go through the same padding path as the stereo
    images. If they do not, the initialization becomes spatially misaligned
    inside the model and the evaluation is no longer meaningful.
    """
    image1, image2 = padder.pad(image1, image2)
    if use_lidar:
        # The external initialization must be padded exactly like the images,
        # otherwise it will be spatially misaligned inside the model.
        external_disp, external_valid = padder.pad(external_disp, external_valid)
        kwargs = dict(
            external_disp=external_disp,
            external_valid=external_valid,
            lidar_fill_dav2=lidar_fill_dav2,
        )
    else:
        kwargs = dict()

    with autocast(enabled=mixed_prec):
        disp_pr = model(
            image1,
            image2,
            iters=iters,
            scale_iters=scale_iters,
            test_mode=True,
            **kwargs,
        )

    disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)

    return disp_pr


@torch.no_grad()
def validate_middlebury_lidar(model, args, use_lidar=True, tag="lidar"):
    """Evaluate one full Middlebury split with either LiDAR or DAV2 init.

    Args:
        model: Initialized DEFOM-Stereo model on CUDA.
        args: Parsed command line arguments.
        use_lidar: If ``True``, inject external LiDAR disparity. If ``False``,
            run the same scenes with the model's internal DAV2 initialization.
        tag: Name used in logs and returned metric keys.

    Returns:
        A dictionary containing average EPE, bad-pixel rate, and FPS.

    The function first computes per-scene metrics, then averages them across
    the evaluated split. This matches the style used by the original stereo
    evaluation script and makes LiDAR-init vs DAV2-init comparisons fair.
    """
    model.eval()
    val_dataset = build_eval_dataset(args)
    fill_dav2 = args.lidar_fill == "dav2"
    mode_str = "LiDAR init" if use_lidar else "DAV2 init"

    out_list, epe_list, elapsed_list = [], [], []
    missing_lidar = []

    for val_id in tqdm(range(len(val_dataset)), desc=f"Middlebury ({tag})"):
        # ``StereoDataset`` returns a dictionary with image tensors, disparity
        # ground truth, valid mask, and original file paths.
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]
        image1_path = data_blob["imageL_file"]

        external_disp, external_valid = None, None
        if use_lidar:
            try:
                external_disp, external_valid, lidar_path = load_lidar_batch(
                    image1, image1_path, args, image1.device
                )
            except FileNotFoundError as e:
                missing_lidar.append(str(e))
                logging.warning(str(e))
                continue
            if args.verbose:
                logging.info("scene %s lidar: %s", image1_path, lidar_path)

        # DEFOM expects sizes divisible by 32 in this evaluation configuration.
        padder = InputPadder(image1.shape, divis_by=32)

        start = time.time()
        disp_pr = run_inference(
            model,
            image1,
            image2,
            padder,
            external_disp,
            external_valid,
            args.valid_iters,
            args.scale_iters,
            args.mixed_precision,
            use_lidar,
            fill_dav2,
        )
        elapsed_list.append(time.time() - start)

        # The dataset stores disparity as ``(1, H, W)``, so we keep the same
        # layout for the prediction and compute pixelwise absolute error.
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.sum(torch.abs(disp_pr - disp_gt), dim=0)
        epe_flattened = epe.flatten()
        val = (valid.reshape(-1) >= 0.5) & (disp_gt.reshape(-1) < args.max_disp)

        out = epe_flattened > args.bad_threshold
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            "Middlebury %s [%s] %d/%d  EPE %.4f  Out%.1f %.4f  %.3fs",
            tag,
            mode_str,
            val_id + 1,
            len(val_dataset),
            image_epe,
            args.bad_threshold,
            image_out,
            elapsed_list[-1],
        )
        epe_list.append(image_epe)
        out_list.append(image_out)

    if missing_lidar:
        logging.warning("Skipped %d scenes (missing LiDAR PNG).", len(missing_lidar))

    if len(epe_list) == 0:
        raise RuntimeError("No scenes evaluated. Check paths and --lidar_disp_name.")

    # Report split-level summary after averaging per-scene metrics.
    epe = float(np.mean(epe_list))
    out2 = 100 * float(np.mean(out_list))
    avg_runtime = float(np.mean(elapsed_list))
    print(
        f"Validation Middlebury ({tag}, {mode_str}): EPE {epe:.4f}, "
        f"Out{args.bad_threshold:g} {out2:.4f}, "
        f"{1/avg_runtime:.2f} FPS ({avg_runtime:.3f}s)"
    )
    return {
        f"middlebury-{tag}-epe": epe,
        f"middlebury-{tag}-out{args.bad_threshold:g}": out2,
        f"middlebury-{tag}-fps": 1 / avg_runtime,
    }


if __name__ == "__main__":
    # Arguments are grouped roughly by:
    # 1) dataset and input paths
    # 2) evaluation behavior
    # 3) model architecture settings needed to load the checkpoint correctly
    parser = argparse.ArgumentParser(
        description="Middlebury eval with dense LiDAR disparity initialization (Plan 1)"
    )
    parser.add_argument(
        "--restore_ckpt", required=True, help="DEFOM-Stereo checkpoint (.pth)"
    )
    parser.add_argument(
        "--middlebury_root",
        type=str,
        default="./datasets/Middlebury",
        help="Middlebury root or flat scene parent (with --split custom)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="2014",
        choices=["F", "H", "Q", "2014", "2021", "custom"],
        help="Middlebury split; custom = root/*/im0.png",
    )
    parser.add_argument(
        "--lidar_root",
        type=str,
        default=None,
        help="Optional root: lidar at lidar_root/<scene>/<lidar_disp_name>",
    )
    parser.add_argument(
        "--lidar_disp_name",
        type=str,
        default="disp0_lidar_dense.png",
        help="LiDAR disparity PNG in each scene folder",
    )
    parser.add_argument(
        "--gt_disp_name",
        type=str,
        default="disp0GT.pfm",
        help="GT filename for --split custom (also tries disp0_gt.png)",
    )
    parser.add_argument(
        "--lidar_fill",
        type=str,
        default="dav2",
        choices=["dav2", "const"],
        help="Invalid LiDAR pixels: fill with DAV2 init or constant 0.01",
    )
    parser.add_argument(
        "--compare_dav2",
        action="store_true",
        help="Also run DAV2-only init on the same scenes for comparison",
    )
    parser.add_argument("--max_disp", type=float, default=1000.0)
    parser.add_argument("--bad_threshold", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--scale_iters", type=int, default=8)

    parser.add_argument(
        "--dinov2_encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
    )
    parser.add_argument("--idepth_scale", type=float, default=0.5)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)
    parser.add_argument(
        "--corr_implementation",
        choices=["reg", "alt", "reg_cuda", "alt_cuda"],
        default="reg",
    )
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument(
        "--scale_list",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    parser.add_argument("--scale_corr_radius", type=int, default=2)
    parser.add_argument("--n_downsample", type=int, default=2, choices=[2, 3])
    parser.add_argument(
        "--context_norm",
        type=str,
        default="batch",
        choices=["group", "batch", "instance", "none"],
    )
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument(
        "--extractor_module",
        type=str,
        default="extractor",
        choices=["extractor", "extractor_defom"],
        help="which extractor implementation to build into DEFOM-Stereo",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    # Build the model with the same architectural options that were used when
    # the checkpoint was trained or exported. This is important because the
    # checkpoint is not self-describing for every architecture choice.
    model = DEFOMStereo(args)
    logging.info("Loading checkpoint: %s", args.restore_ckpt)
    checkpoint = torch.load(args.restore_ckpt, map_location="cuda")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    n_params, n_train = count_parameters(model)
    print(
        f"The model has {n_params/1e6:.2f}M parameters ({n_train/1e6:.2f}M trainable)."
    )

    # CUDA correlation backends are stable enough for full mixed precision, so
    # we enable it automatically when one of those backends is selected.
    use_mixed_precision = (
        args.corr_implementation.endswith("_cuda") or args.mixed_precision
    )
    args.mixed_precision = use_mixed_precision

    results = {}
    # Always run the LiDAR-initialized path first because it is the main target
    # of this script.
    results.update(validate_middlebury_lidar(model, args, use_lidar=True, tag="lidar"))

    if args.compare_dav2:
        results.update(
            validate_middlebury_lidar(model, args, use_lidar=False, tag="dav2")
        )
        print("\n--- Comparison ---")
        print(f"  LiDAR init EPE: {results['middlebury-lidar-epe']:.4f}")
        print(f"  DAV2 init EPE:  {results['middlebury-dav2-epe']:.4f}")
