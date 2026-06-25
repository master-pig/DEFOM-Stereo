import argparse
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(".."))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from core.defom_stereo import DEFOMStereo, autocast
from core.utils.utils import InputPadder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_matrix_line(line, rows, cols):
    values = np.fromstring(line.strip(), sep=" ", dtype=np.float64)
    expected = rows * cols
    if values.size != expected:
        raise ValueError(f"Expected {expected} values, got {values.size}: {line!r}")
    return values.reshape(rows, cols)


def parse_vector_line(line, size):
    values = np.fromstring(line.strip(), sep=" ", dtype=np.float64)
    if values.size != size:
        raise ValueError(f"Expected {size} values, got {values.size}: {line!r}")
    return values


def load_stereo_params(stereo_cam_dir):
    cam0_lines = (stereo_cam_dir / "cam0.txt").read_text(encoding="utf-8").splitlines()
    cam1_lines = (stereo_cam_dir / "cam1.txt").read_text(encoding="utf-8").splitlines()
    camrt_lines = (stereo_cam_dir / "camrt.txt").read_text(encoding="utf-8").splitlines()

    k1 = parse_matrix_line(cam0_lines[0], 3, 3)
    d1 = np.fromstring(cam0_lines[1].strip(), sep=" ", dtype=np.float64)
    k2 = parse_matrix_line(cam1_lines[0], 3, 3)
    d2 = np.fromstring(cam1_lines[1].strip(), sep=" ", dtype=np.float64)
    r = parse_matrix_line(camrt_lines[0], 3, 3)
    t = parse_vector_line(camrt_lines[1], 3).reshape(3, 1)

    # Current tube calibration stores only a few radial coefficients.
    d1_cv = np.zeros((5, 1), dtype=np.float64)
    d2_cv = np.zeros((5, 1), dtype=np.float64)
    d1_cv[: min(5, d1.size), 0] = d1[:5]
    d2_cv[: min(5, d2.size), 0] = d2[:5]

    return k1, d1_cv, k2, d2_cv, r, t


def build_rectification(stereo_cam_dir, image_size):
    k1, d1, k2, d2, r, t = load_stereo_params(stereo_cam_dir)
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        cameraMatrix1=k1,
        distCoeffs1=d1,
        cameraMatrix2=k2,
        distCoeffs2=d2,
        imageSize=image_size,
        R=r,
        T=t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        k1, d1, r1, p1, image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        k2, d2, r2, p2, image_size, cv2.CV_32FC1
    )
    return (map1x, map1y), (map2x, map2y), q


def rectify_pair(left_rgb, right_rgb, left_maps, right_maps):
    map1x, map1y = left_maps
    map2x, map2y = right_maps
    left_rect = cv2.remap(left_rgb, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_rgb, map2x, map2y, interpolation=cv2.INTER_LINEAR)
    return left_rect, right_rect


def image_to_tensor(image_rgb):
    image = np.ascontiguousarray(image_rgb)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return tensor[None].to(DEVICE)


def resize_for_inference(image_left, image_right, resize_width, resize_height):
    if resize_width is None or resize_height is None:
        return image_left, image_right
    left_resized = cv2.resize(image_left, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    right_resized = cv2.resize(image_right, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    return left_resized, right_resized


def run_model(model, left_rect, right_rect, args):
    orig_h, orig_w = left_rect.shape[:2]
    infer_left, infer_right = resize_for_inference(
        left_rect, right_rect, args.resize_width, args.resize_height
    )

    image1 = image_to_tensor(infer_left)
    image2 = image_to_tensor(infer_right)
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        with autocast(enabled=args.mixed_precision):
            disp_pr = model(
                image1,
                image2,
                iters=args.valid_iters,
                scale_iters=args.scale_iters,
                test_mode=True,
            )

    disp_pr = padder.unpad(disp_pr).float()
    infer_h, infer_w = disp_pr.shape[-2:]

    if (infer_h, infer_w) != (orig_h, orig_w):
        scale_x = orig_w / infer_w
        disp_pr = F.interpolate(disp_pr, size=(orig_h, orig_w), mode="bilinear", align_corners=True)
        disp_pr = disp_pr * scale_x

    return disp_pr.squeeze().detach().cpu().numpy().astype(np.float32)


def save_disparity_outputs(disparity, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{stem}_disp.npy", disparity)

    disp_u16 = np.clip(np.round(np.maximum(disparity, 0.0) * 256.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(output_dir / f"{stem}_disp_u16.png"), disp_u16)

    finite_mask = np.isfinite(disparity)
    if finite_mask.any():
        valid_disp = disparity[finite_mask]
        vmin = float(valid_disp.min())
        vmax = float(valid_disp.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        plt.imsave(output_dir / f"{stem}_disp_vis.png", disparity, cmap="jet", vmin=vmin, vmax=vmax)
    else:
        plt.imsave(output_dir / f"{stem}_disp_vis.png", np.zeros_like(disparity), cmap="jet")


def disparity_to_pointcloud(disparity, left_rect, q, min_disp):
    points_3d = cv2.reprojectImageTo3D(disparity, q)
    mask = np.isfinite(points_3d).all(axis=2) & np.isfinite(disparity) & (disparity > min_disp)
    points = points_3d[mask]
    colors = left_rect[mask]
    return points, colors


def save_ply(points_xyz, colors_rgb, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def build_model(args):
    model = DEFOMStereo(args)
    checkpoint = torch.load(args.restore_ckpt, map_location=DEVICE)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def collect_pairs(date_dir):
    left_dir = date_dir / "left"
    right_dir = date_dir / "right"
    left_files = {path.name: path for path in left_dir.glob("*.BMP")}
    right_files = {path.name: path for path in right_dir.glob("*.BMP")}
    names = sorted(set(left_files) & set(right_files))
    if not names:
        raise FileNotFoundError(f"No matching BMP pairs found in {left_dir} and {right_dir}")
    return [(left_files[name], right_files[name]) for name in names]


def process_tube_date(args):
    date_dir = Path(args.tube_root) / args.tube_date
    stereo_cam_dir = date_dir / "stereo_cam"
    result_dir = date_dir / "result"
    disparity_dir = result_dir / "disparity"
    pointcloud_dir = result_dir / "pointcloud"

    pairs = collect_pairs(date_dir)
    sample_bgr = cv2.imread(str(pairs[0][0]), cv2.IMREAD_COLOR)
    if sample_bgr is None:
        raise FileNotFoundError(f"Failed to read sample image: {pairs[0][0]}")

    image_size = (sample_bgr.shape[1], sample_bgr.shape[0])
    left_maps, right_maps, q = build_rectification(stereo_cam_dir, image_size)
    model = build_model(args)

    print(f"Found {len(pairs)} stereo pairs under {date_dir}")
    print(f"Saving outputs to {result_dir}")

    for left_path, right_path in pairs:
        stem = left_path.stem
        print(f"[{stem}] rectifying")
        left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_bgr is None or right_bgr is None:
            raise FileNotFoundError(f"Failed to read stereo pair: {left_path}, {right_path}")

        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        left_rect, right_rect = rectify_pair(left_rgb, right_rgb, left_maps, right_maps)

        print(f"[{stem}] estimating disparity")
        disparity = run_model(model, left_rect, right_rect, args)
        save_disparity_outputs(disparity, disparity_dir, stem)

        print(f"[{stem}] exporting point cloud")
        points_xyz, colors_rgb = disparity_to_pointcloud(disparity, left_rect, q, args.min_disp)
        save_ply(points_xyz, colors_rgb, pointcloud_dir / f"{stem}.ply")


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run DEFOM-Stereo on tube stereo images and export disparity + point cloud."
    )
    parser.add_argument("--tube-root", type=str, default="tube")
    parser.add_argument("--tube-date", type=str, required=True)
    parser.add_argument("--restore-ckpt", type=str, required=True, help="checkpoint path")
    parser.add_argument(
        "--dinov2-encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="encoder type matching the checkpoint",
    )
    parser.add_argument("--resize-width", type=int, default=None, help="inference width after rectification")
    parser.add_argument("--resize-height", type=int, default=None, help="inference height after rectification")
    parser.add_argument("--min-disp", type=float, default=0.1)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--scale-iters", type=int, default=8)

    parser.add_argument("--idepth-scale", type=float, default=0.5)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument("--corr-implementation", choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg")
    parser.add_argument("--shared-backbone", action="store_true")
    parser.add_argument("--corr-levels", type=int, default=2)
    parser.add_argument("--corr-radius", type=int, default=4)
    parser.add_argument(
        "--scale-list",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    parser.add_argument("--scale-corr-radius", type=int, default=2)
    parser.add_argument("--n-downsample", type=int, default=2, choices=[2, 3])
    parser.add_argument("--context-norm", type=str, default="batch", choices=["group", "batch", "instance", "none"])
    parser.add_argument("--n-gru-layers", type=int, default=3)
    parser.add_argument("--extractor-module", type=str, default="extractor", choices=["extractor", "extractor_defom"])
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    process_tube_date(args)
