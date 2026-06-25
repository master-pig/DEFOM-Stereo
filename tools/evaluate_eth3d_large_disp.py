from __future__ import print_function, division

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.defom_stereo import DEFOMStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from evaluate_stereo import count_parameters


def save_array_and_vis(path_prefix, array, cmap="jet"):
    np.save(f"{path_prefix}.npy", array)
    plt.imsave(f"{path_prefix}.png", array, cmap=cmap)


@torch.no_grad()
def validate_eth3d_large_disp(model, args, mixed_prec=False):
    model.eval()
    val_dataset = datasets.ETH3D({}, is_eval=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    flagged = 0
    for val_id in tqdm(range(len(val_dataset)), desc="ETH3D large-disp scan"):
        data_blob = val_dataset[val_id]
        image1 = data_blob["img1"][None].cuda()
        image2 = data_blob["img2"][None].cuda()
        disp_gt = data_blob["disp"]
        valid = data_blob["valid"]
        imageL_file = data_blob["imageL_file"]
        scene_name = Path(imageL_file).parent.name

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            disp_pr = model(
                image1,
                image2,
                iters=args.valid_iters,
                scale_iters=args.scale_iters,
                test_mode=True,
            )

        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0).squeeze(0)
        disp_gt = disp_gt.squeeze(0).cpu()
        valid = valid.squeeze(0).cpu() >= 0.5

        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.abs(disp_pr - disp_gt)

        valid_disp = disp_pr[valid]
        if valid_disp.numel() == 0:
            continue

        pred_max = float(valid_disp.max().item())
        pred_median = float(valid_disp.median().item())
        gt_valid = disp_gt[valid]
        gt_median = float(gt_valid.median().item())
        epe_mean = float(epe[valid].mean().item())
        out1 = float((epe[valid] > 1.0).float().mean().item())

        if pred_max <= args.pred_threshold:
            continue

        flagged += 1
        summary_rows.append(
            {
                "scene": scene_name,
                "imageL_file": imageL_file,
                "pred_max": pred_max,
                "pred_median": pred_median,
                "gt_median": gt_median,
                "median_ratio": pred_median / max(gt_median, 1e-6),
                "epe_mean": epe_mean,
                "out1": out1,
                "valid_pixels": int(valid.sum().item()),
                "pixels_over_threshold": int(((disp_pr > args.pred_threshold) & valid).sum().item()),
            }
        )

        prefix = output_dir / scene_name
        pred_np = disp_pr.numpy()
        gt_np = disp_gt.numpy()
        epe_np = epe.numpy()
        valid_np = valid.numpy().astype(np.uint8)
        pred_large_np = ((pred_np > args.pred_threshold) & (valid_np > 0)).astype(np.uint8)

        pred_vis = np.where(valid_np > 0, pred_np, np.nan)
        gt_vis = np.where(valid_np > 0, gt_np, np.nan)
        epe_vis = np.where(valid_np > 0, epe_np, np.nan)

        save_array_and_vis(prefix.with_name(prefix.name + "_pred"), pred_np, cmap="jet")
        save_array_and_vis(prefix.with_name(prefix.name + "_pred_vis"), pred_vis, cmap="jet")
        save_array_and_vis(prefix.with_name(prefix.name + "_gt"), gt_vis, cmap="jet")
        save_array_and_vis(prefix.with_name(prefix.name + "_epe"), epe_vis, cmap="inferno")
        save_array_and_vis(prefix.with_name(prefix.name + "_valid"), valid_np, cmap="gray")
        save_array_and_vis(prefix.with_name(prefix.name + "_pred_over_threshold"), pred_large_np, cmap="gray")

        logging.info(
            "Flagged ETH3D sample %s: pred_max=%.4f pred_median=%.4f gt_median=%.4f ratio=%.4f epe=%.4f out1=%.4f",
            scene_name,
            pred_max,
            pred_median,
            gt_median,
            pred_median / max(gt_median, 1e-6),
            epe_mean,
            out1,
        )

    summary_path = output_dir / "flagged_samples.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "imageL_file",
                "pred_max",
                "pred_median",
                "gt_median",
                "median_ratio",
                "epe_mean",
                "out1",
                "valid_pixels",
                "pixels_over_threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"Flagged {flagged} / {len(val_dataset)} ETH3D samples with pred_max > {args.pred_threshold}. "
        f"Saved to {summary_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ETH3D and export samples whose predicted disparity exceeds a threshold."
    )
    parser.add_argument("--restore_ckpt", required=True, help="restore checkpoint")
    parser.add_argument("--output_dir", type=str, default="eth3d_large_disp", help="directory to save flagged samples")
    parser.add_argument("--pred_threshold", type=float, default=100.0, help="flag sample if valid predicted disparity exceeds this value")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--scale_iters", type=int, default=8)

    parser.add_argument("--dinov2_encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--idepth_scale", type=float, default=0.5)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)
    parser.add_argument("--corr_implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
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
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"])
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
    print(f"The model has {n_params/1e6:.2f}M parameters ({n_train/1e6:.2f}M trainable).")

    use_mixed_precision = args.corr_implementation.endswith("_cuda") or args.mixed_precision
    validate_eth3d_large_disp(model, args, mixed_prec=use_mixed_precision)
