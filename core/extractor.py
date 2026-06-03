import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from depth_anything_v2.dpt import DepthAnythingV2


class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

    def forward(self, x):

        return self.relu(self.norm1(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1, ratio=4):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // ratio, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            planes // ratio, planes // ratio, kernel_size=3, padding=1, stride=stride
        )
        self.conv3 = nn.Conv2d(planes // ratio, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes // ratio
            )
            self.norm2 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes // ratio
            )
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes // ratio)
            self.norm2 = nn.BatchNorm2d(planes // ratio)
            self.norm3 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes // ratio)
            self.norm2 = nn.InstanceNorm2d(planes // ratio)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm4 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class DeformConv2dBlock(nn.Module):
    def __init__(
        self, in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True
    ):
        super(DeformConv2dBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.weight = nn.Parameter(torch.empty(planes, in_planes, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(planes))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def _build_sampling_grid(self, x, offset):
        b, _, h, w = x.shape
        _, _, h_out, w_out = offset.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        dtype = x.dtype
        device = x.device

        y_base = torch.arange(h_out, device=device, dtype=dtype) * sh - ph
        x_base = torch.arange(w_out, device=device, dtype=dtype) * sw - pw
        grid_y, grid_x = torch.meshgrid(y_base, x_base, indexing="ij")

        kernel_y = torch.arange(kh, device=device, dtype=dtype)
        kernel_x = torch.arange(kw, device=device, dtype=dtype)
        kernel_grid_y, kernel_grid_x = torch.meshgrid(kernel_y, kernel_x, indexing="ij")

        grid_y = grid_y[None, :, :, None] + kernel_grid_y.reshape(1, 1, 1, -1)
        grid_x = grid_x[None, :, :, None] + kernel_grid_x.reshape(1, 1, 1, -1)

        offset = offset.view(b, kh * kw, 2, h_out, w_out).permute(0, 3, 4, 1, 2)
        sample_y = grid_y + offset[..., 0]
        sample_x = grid_x + offset[..., 1]

        if h > 1:
            sample_y = 2.0 * sample_y / (h - 1) - 1.0
        else:
            sample_y = torch.zeros_like(sample_y)

        if w > 1:
            sample_x = 2.0 * sample_x / (w - 1) - 1.0
        else:
            sample_x = torch.zeros_like(sample_x)

        grid = torch.stack((sample_x, sample_y), dim=-1)
        return grid.reshape(b, h_out, w_out * kh * kw, 2)

    def forward(self, x, offset, mask=None):
        kh, kw = self.kernel_size
        h_out, w_out = offset.shape[-2:]

        grid = self._build_sampling_grid(x, offset)
        sampled = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.view(x.shape[0], x.shape[1], h_out, w_out, kh * kw).permute(
            0, 1, 4, 2, 3
        )

        if mask is not None:
            sampled = sampled * mask.unsqueeze(1)

        weight = self.weight.view(self.planes, self.in_planes, kh * kw)
        out = torch.einsum("ock,bckhw->bohw", weight, sampled)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out


class BasicEncoder(nn.Module):
    def __init__(self, d_dim, output_dim=128, norm_fn="batch", downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample
        self.offset_range = 2.0
        stem_stride = 1 + (downsample > 2)

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1_std = nn.Conv2d(3, 64, kernel_size=7, stride=stem_stride, padding=3)
        self.conv1_def = DeformConv2dBlock(
            3, 64, kernel_size=7, stride=stem_stride, padding=3
        )
        self.guide_reduce = ConvBlock(d_dim, 64, self.norm_fn)
        self.guide_refine = ResidualBlock(64, 64, self.norm_fn)
        self.offset_head = nn.Conv2d(64, 7 * 7 * 2, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(64, 7 * 7, kernel_size=3, padding=1)
        self.alpha_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))

        # depth feat convolution
        self.convd = ConvBlock(d_dim, 128, self.norm_fn)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.offset_head.weight, 0)
        nn.init.constant_(self.offset_head.bias, 0)
        nn.init.constant_(self.mask_head.weight, 0)
        nn.init.constant_(self.mask_head.bias, 2.0)
        nn.init.constant_(self.alpha_head.weight, 0)
        nn.init.constant_(self.alpha_head.bias, -2.0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, dfeats):

        # if input is list, combine batch dimension
        x_is_list = isinstance(x, tuple) or isinstance(x, list)
        if x_is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        dfeats_is_list = isinstance(dfeats, tuple) or isinstance(dfeats, list)
        if dfeats_is_list:
            batch_dim = dfeats[0].shape[0]
            dfeats = torch.cat(dfeats, dim=0)

        guide = self.guide_reduce(dfeats)
        guide = self.guide_refine(guide)

        stem_std = self.conv1_std(x)
        guide = F.interpolate(
            guide, stem_std.shape[-2:], mode="bilinear", align_corners=True
        )

        offset = self.offset_range * torch.tanh(self.offset_head(guide))
        mask = torch.sigmoid(self.mask_head(guide))
        alpha = torch.sigmoid(self.alpha_head(guide))

        stem_def = self.conv1_def(x, offset, mask)
        x = stem_std + alpha * stem_def
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x + self.convd(dfeats)

        x = self.conv2(x)

        if x_is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x


class MultiBasicEncoder(nn.Module):
    def __init__(
        self,
        d_dim,
        output_dim=[128, 128, 128],
        norm_fn="batch",
        downsample=3,
        drop_path_rate=0.2,
    ):
        super(MultiBasicEncoder, self).__init__()
        self.d_dim = d_dim
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        self.drop_path = DropPath(drop_path_rate)

        self.conv08 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1),
            )
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

        self.conv16 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1),
            )
            output_list.append(conv_out)

        self.outputs16 = nn.ModuleList(output_list)

        self.conv32 = ConvBlock(d_dim, 128, self.norm_fn)
        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)

        self.outputs32 = nn.ModuleList(output_list)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, d_feats, num_layers=3):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        feat = x + self.drop_path(self.conv08(d_feats[0]))
        outputs08 = [f(feat) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08,)

        y = self.layer4(x)
        feat = y + self.drop_path(self.conv16(d_feats[1]))
        outputs16 = [f(feat) for f in self.outputs16]

        if num_layers == 2:
            return (outputs08, outputs16)

        z = self.layer5(y)
        feat = z + self.drop_path(self.conv32(d_feats[2]))
        outputs32 = [f(feat) for f in self.outputs32]

        return (outputs08, outputs16, outputs32)


class DefomEncoder(nn.Module):
    def __init__(self, dinov2_encoder, pretrained=True, freeze=True, idepth_scale=0.25):
        super(DefomEncoder, self).__init__()
        self.dinov2_encoder = dinov2_encoder
        self.idepth_scale = idepth_scale
        self.pretrained = pretrained
        self.freeze = freeze

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }

        self.depth_anything = DepthAnythingV2(**model_configs[self.dinov2_encoder])

        if pretrained and os.path.exists(
            f"./checkpoints/depth_anything_v2_{dinov2_encoder}.pth"
        ):
            self.depth_anything.load_state_dict(
                torch.load(
                    f"./checkpoints/depth_anything_v2_{dinov2_encoder}.pth",
                    map_location="cpu",
                ),
                strict=False,
            )
        if freeze:
            for param in self.depth_anything.pretrained.parameters():
                param.requires_grad = False
            for param in self.depth_anything.depth_head.parameters():
                param.requires_grad = False

        self.out_dim = model_configs[self.dinov2_encoder]["features"]

    def _dav2_init_disp(self, idepth, ow):
        bs = idepth.shape[0]
        max_idepth, _ = torch.max(idepth.view(bs, -1), dim=1)
        max_idepth = max_idepth.detach().view(bs, 1, 1, 1) + 1e-8
        return idepth / max_idepth * self.idepth_scale * ow + 0.01

    def forward(
        self,
        x,
        danv2_io_sizes,
        external_disp=None,
        external_valid=None,
        lidar_fill_dav2=True,
    ):

        x = torch.cat(x, dim=0)
        ih, iw, oh, ow = danv2_io_sizes
        x = F.interpolate(x, (ih, iw), mode="bilinear", align_corners=True)

        features, left_feat, right_feat, idepth = self.depth_anything(x, oh, ow)

        dav2_disp = self._dav2_init_disp(idepth, ow)

        if external_disp is None:
            return features, left_feat, right_feat, dav2_disp

        from core.utils.lidar_disp import prepare_init_disp

        if external_valid is None:
            external_valid = (external_disp > 1e-6).float()
        disp_init = prepare_init_disp(
            external_disp,
            external_valid,
            danv2_io_sizes,
            dav2_disp=dav2_disp,
            fill_invalid_with_dav2=lidar_fill_dav2,
        )
        return features, left_feat, right_feat, disp_init
