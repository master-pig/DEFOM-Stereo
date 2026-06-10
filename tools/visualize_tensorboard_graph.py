import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TensorBoardGraphWrapper(nn.Module):
    def __init__(self, model, iters=4, scale_iters=2):
        super().__init__()
        self.model = model
        self.iters = iters
        self.scale_iters = scale_iters

    def forward(self, image1, image2):
        return self.model(
            image1,
            image2,
            iters=self.iters,
            scale_iters=self.scale_iters,
            test_mode=True,
        )


def build_model_args(cli_args):
    return argparse.Namespace(
        dinov2_encoder=cli_args.dinov2_encoder,
        idepth_scale=cli_args.idepth_scale,
        mixed_precision=False,
        n_downsample=cli_args.n_downsample,
        context_norm=cli_args.context_norm,
        n_gru_layers=cli_args.n_gru_layers,
        hidden_dims=[cli_args.hidden_dim] * cli_args.n_gru_layers,
        corr_radius=cli_args.corr_radius,
        corr_levels=cli_args.corr_levels,
        scale_list=cli_args.scale_list,
        scale_corr_radius=cli_args.scale_corr_radius,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the current DEFOM-Stereo network graph to TensorBoard."
    )
    parser.add_argument("--logdir", type=str, default="runs/tensorboard_graph/defom_stereo_current")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=736)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--scale_iters", type=int, default=2)
    parser.add_argument("--dinov2_encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--idepth_scale", type=float, default=0.5)
    parser.add_argument("--n_downsample", type=int, default=2, choices=[2, 3])
    parser.add_argument("--context_norm", type=str, default="batch", choices=["group", "batch", "instance", "none"])
    parser.add_argument("--n_gru_layers", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--corr_levels", type=int, default=2)
    parser.add_argument("--scale_corr_radius", type=int, default=2)
    parser.add_argument(
        "--scale_list",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    )
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or leave empty for auto")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cli_args = parse_args()
    device = resolve_device(cli_args.device)

    from core.defom_stereo import DEFOMStereo

    model_args = build_model_args(cli_args)
    base_model = DEFOMStereo(model_args).to(device)
    base_model.eval()

    model = TensorBoardGraphWrapper(
        base_model,
        iters=cli_args.iters,
        scale_iters=cli_args.scale_iters,
    ).to(device)
    model.eval()

    image1 = torch.randint(
        0,
        256,
        (cli_args.batch_size, 3, cli_args.height, cli_args.width),
        dtype=torch.float32,
        device=device,
    )
    image2 = torch.randint(
        0,
        256,
        (cli_args.batch_size, 3, cli_args.height, cli_args.width),
        dtype=torch.float32,
        device=device,
    )

    logdir = Path(cli_args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(logdir))
    with torch.no_grad():
        writer.add_graph(model, (image1, image2))
    writer.close()

    print(f"TensorBoard graph saved to: {logdir.resolve()}")
    print(f"Start TensorBoard with: tensorboard --logdir {logdir.parent}")


if __name__ == "__main__":
    main()
