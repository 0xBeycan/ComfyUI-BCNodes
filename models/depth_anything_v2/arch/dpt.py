# Depth Anything V2 — Licensed under the Apache License, Version 2.0 (LICENSE in this directory).
# https://github.com/DepthAnything/Depth-Anything-V2
#
# Vendored from Depth Anything V2 (depth_anything_v2/dpt.py), inference path of the ViT-S model
# (Depth-Anything-V2-Small, the Apache-2.0 checkpoint) only: the encoder is fixed to "vits" with
# its published head config (features 64, out_channels [48, 96, 192, 384], no BN, no class
# token readout, blocks [2, 5, 8, 11]). infer_image / image2tensor (cv2 + torchvision
# preprocessing) are dropped; ../inference.py does the same preprocessing in torch. The unused
# ConvBlock and the class-token readout branch are dropped. Parameter names are unchanged, so
# depth_anything_v2_vits.pth loads strict.

import torch.nn as nn
import torch.nn.functional as F

from .blocks import FeatureFusionBlock, _make_scratch
from .dinov2 import DINOv2_vits

INTERMEDIATE_LAYERS = [2, 5, 8, 11]


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out


class DepthAnythingV2(nn.Module):
    """Depth-Anything-V2-Small: (B, 3, H, W) ImageNet-normalised, H and W multiples of 14 ->
    (B, H, W) relative inverse depth (larger = nearer), >= 0."""

    def __init__(
        self,
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
    ):
        super(DepthAnythingV2, self).__init__()

        self.pretrained = DINOv2_vits()

        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels)

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(x, INTERMEDIATE_LAYERS, return_class_token=True)

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)

        return depth.squeeze(1)
