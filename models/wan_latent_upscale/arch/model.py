# Vendored from spacepxl's Wan 2.1 latent upscaler (MIT License, see LICENSE in this directory).
# Kept as released, minus the unused weight init and the commented-out layers: the module and
# parameter names are the weight file's keys.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm3d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class DiCoBlock3d(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0, kernel_size=3):
        super().__init__()
        self.conv2 = nn.Conv3d(
            hidden_size,
            hidden_size,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, kernel_size // 2, kernel_size // 2),
            padding_mode="replicate",
        )
        self.conv3 = nn.Conv3d(hidden_size, hidden_size, kernel_size=1)
        self.cca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=1),
            nn.Sigmoid(),
        )
        ffn_channel = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv3d(hidden_size, ffn_channel, kernel_size=1)
        self.conv5 = nn.Conv3d(ffn_channel, hidden_size, kernel_size=1)
        self.norm1 = LayerNorm3d(hidden_size, affine=False)
        self.norm2 = LayerNorm3d(hidden_size, affine=False)

    def forward(self, inp):
        x = self.norm1(inp)
        x = F.gelu(self.conv2(x))
        x = self.conv3(x * self.cca(x))
        x = inp + x

        y = self.norm2(x)
        y = self.conv5(F.gelu(self.conv4(y)))
        return x + y


class LatentModel3d(nn.Module):
    def __init__(
        self,
        in_channels=16,
        out_channels=16,
        hidden_size=128,
        kernel_size=3,
        mlp_ratio=4.0,
        depth=8,
        upscale=2,
    ):
        super().__init__()
        self.upscale = int(upscale)
        self.conv_in = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )
        self.blocks = nn.ModuleList([
            DiCoBlock3d(hidden_size, mlp_ratio, kernel_size)
            for _ in range(depth)
        ])
        out_ch = out_channels * self.upscale ** 2
        self.conv_out = nn.Conv3d(
            hidden_size,
            out_ch,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            padding_mode="replicate",
        )

    def forward(self, x):
        """
        x: (B, C, F, H, W) tensor of spatial inputs (images or latent representations of images)
        """
        hidden = self.conv_in(x)
        for block in self.blocks:
            hidden = block(hidden)
        output = self.conv_out(hidden)
        if self.upscale > 1:
            output = F.pixel_shuffle(output.movedim(1, 2), self.upscale).movedim(2, 1)
            x = F.interpolate(x, scale_factor=(1, self.upscale, self.upscale), mode="nearest-exact")
        return output + x
