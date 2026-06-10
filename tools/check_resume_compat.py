import argparse

import torch

from core.defom_stereo import DEFOMStereo


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--dinov2_encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
    )
    parser.add_argument("--idepth_scale", type=float, default=0.5)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--n_downsample", type=int, default=2, choices=[2, 3])
    parser.add_argument(
        "--context_norm",
        type=str,
        default="batch",
        choices=["group", "batch", "instance", "none"],
    )
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 128, 128])
    parser.add_argument(
        "--corr_implementation",
        choices=["reg", "alt", "reg_cuda", "alt_cuda"],
        default="reg",
    )
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument(
        "--scale_list",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    parser.add_argument("--scale_corr_radius", type=int, default=2)
    return parser.parse_args()


def main():
    args = build_args()

    model = DEFOMStereo(args)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    for name, module in model.named_modules():
        for b_name, b in module.named_buffers(recurse=False):
            print(name)
            print("  [B]", b_name, b.shape)
        # for p_name, p in module.named_parameters(recurse=False):
        # print(name)
        # print("  [P]", p_name, p.shape)

    model_state = model.state_dict()
    loadable_state = {}
    unexpected_raw = []
    shape_mismatch = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_raw.append(key)
            continue

        if model_state[key].shape != value.shape:
            shape_mismatch.append(
                (key, tuple(value.shape), tuple(model_state[key].shape))
            )
            continue

        loadable_state[key] = value

    result = model.load_state_dict(loadable_state, strict=False)

    print("=== Summary ===")
    print(f"checkpoint: {args.ckpt}")
    print(f"model params: {len(model_state)}")
    print(f"ckpt params: {len(state_dict)}")
    print(f"loadable params: {len(loadable_state)}")
    print(f"missing keys after load: {len(result.missing_keys)}")
    print(f"unexpected keys after load: {len(result.unexpected_keys)}")
    print(f"raw unexpected keys: {len(unexpected_raw)}")
    print(f"shape mismatch keys: {len(shape_mismatch)}")

    print("\n=== Missing Keys ===")
    for key in result.missing_keys:
        print(key)

    print("\n=== Raw Unexpected Keys ===")
    for key in unexpected_raw:
        print(key)

    print("\n=== Shape Mismatch Keys ===")
    for key, ckpt_shape, model_shape in shape_mismatch:
        print(f"{key}: ckpt={ckpt_shape}, model={model_shape}")


if __name__ == "__main__":
    main()
