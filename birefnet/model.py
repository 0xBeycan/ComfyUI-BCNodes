# BiRefNet — Bilateral Reference for High-Resolution Dichotomous Image
# Segmentation (Zheng et al., 2024).
#
# Vendored from BiRefNet (models/birefnet.py) — Copyright (c) 2024 Peng Zheng —
# MIT License — https://github.com/ZhengPeng7/BiRefNet
#
# This is the inference path only, hard-wired to the configuration the public
# checkpoints were trained with:
#   mul_scl_ipt='cat', cxt_num=3, squeeze_block='BasicDecBlk_x1',
#   dec_blk='BasicDecBlk', dec_att='ASPPDeformable', dec_ipt=True,
#   dec_ipt_split=True, ms_supervision=True, out_ref=True, refine=''.
# Training-only branches (multi-scale supervision outputs, gradient labels,
# the kornia laplacian) are dropped; the corresponding parameters are kept so
# the released safetensors load with strict=True. The einops rearrange in
# image2patches is replaced by an equivalent view/permute. It is a plain
# nn.Module — no transformers PreTrainedModel.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import BasicDecBlk, BasicLatBlk, SimpleConvs
from .swin_v1 import BACKBONES, LATERAL_CHANNELS


def image2patches(image, patch_ref):
    """einops 'b c (hg h) (wg w) -> b (c hg wg) h w' with the grid taken from patch_ref."""
    b, c, H, W = image.shape
    gh, gw = H // patch_ref.shape[-2], W // patch_ref.shape[-1]
    if H != gh * patch_ref.shape[-2] or W != gw * patch_ref.shape[-1]:
        raise ValueError(f"image {H}x{W} is not a whole multiple of the reference {tuple(patch_ref.shape[-2:])}")
    h, w = H // gh, W // gw
    x = image.view(b, c, gh, h, gw, w).permute(0, 1, 2, 4, 3, 5)
    return x.reshape(b, c * gh * gw, h, w)


def _interp(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        N_dec_ipt = 64
        ic = 64
        # dec_ipt_split: the input image is folded into patches matching each
        # decoder stage, so the channel count grows with the grid size.
        self.ipt_blk5 = SimpleConvs(2 ** 10 * 3, channels[0] // 8, inter_channels=ic)
        self.ipt_blk4 = SimpleConvs(2 ** 8 * 3, channels[0] // 8, inter_channels=ic)
        self.ipt_blk3 = SimpleConvs(2 ** 6 * 3, channels[1] // 8, inter_channels=ic)
        self.ipt_blk2 = SimpleConvs(2 ** 4 * 3, channels[2] // 8, inter_channels=ic)
        self.ipt_blk1 = SimpleConvs(2 ** 0 * 3, channels[3] // 8, inter_channels=ic)

        self.decoder_block4 = BasicDecBlk(channels[0] + channels[0] // 8, channels[1])
        self.decoder_block3 = BasicDecBlk(channels[1] + channels[0] // 8, channels[2])
        self.decoder_block2 = BasicDecBlk(channels[2] + channels[1] // 8, channels[3])
        self.decoder_block1 = BasicDecBlk(channels[3] + channels[2] // 8, channels[3] // 2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3] // 2 + channels[3] // 8, 1, 1, 1, 0))

        self.lateral_block4 = BasicLatBlk(channels[1], channels[1])
        self.lateral_block3 = BasicLatBlk(channels[2], channels[2])
        self.lateral_block2 = BasicLatBlk(channels[3], channels[3])

        # ms_supervision heads: training-only outputs, kept for the state dict.
        self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
        self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
        self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)

        # out_ref: gradient-aware attention. gdt_convs_pred_* are training-only.
        _N = 16
        self.gdt_convs_4 = nn.Sequential(nn.Conv2d(channels[1], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_3 = nn.Sequential(nn.Conv2d(channels[2], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))
        self.gdt_convs_2 = nn.Sequential(nn.Conv2d(channels[3], _N, 3, 1, 1), nn.BatchNorm2d(_N), nn.ReLU(inplace=True))

        self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

        self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    def forward(self, features):
        x, x1, x2, x3, x4 = features

        patches_batch = image2patches(x, patch_ref=x4)
        x4 = torch.cat((x4, self.ipt_blk5(_interp(patches_batch, x4.shape[2:]))), 1)
        p4 = self.decoder_block4(x4)
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        _p4 = _interp(p4, x3.shape[2:])
        _p3 = _p4 + self.lateral_block4(x3)

        patches_batch = image2patches(x, patch_ref=_p3)
        _p3 = torch.cat((_p3, self.ipt_blk4(_interp(patches_batch, x3.shape[2:]))), 1)
        p3 = self.decoder_block3(_p3)
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
        _p3 = _interp(p3, x2.shape[2:])
        _p2 = _p3 + self.lateral_block3(x2)

        patches_batch = image2patches(x, patch_ref=_p2)
        _p2 = torch.cat((_p2, self.ipt_blk3(_interp(patches_batch, x2.shape[2:]))), 1)
        p2 = self.decoder_block2(_p2)
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
        _p2 = _interp(p2, x1.shape[2:])
        _p1 = _p2 + self.lateral_block2(x1)

        patches_batch = image2patches(x, patch_ref=_p1)
        _p1 = torch.cat((_p1, self.ipt_blk2(_interp(patches_batch, x1.shape[2:]))), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = _interp(_p1, x.shape[2:])

        patches_batch = image2patches(x, patch_ref=_p1)
        _p1 = torch.cat((_p1, self.ipt_blk1(_interp(patches_batch, x.shape[2:]))), 1)
        return self.conv_out1(_p1)


class BiRefNet(nn.Module):
    """Returns the final logits map (B, 1, H, W); apply sigmoid for the matte."""

    def __init__(self, backbone="swin_v1_l"):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; expected one of {sorted(BACKBONES)}")
        self.bb = BACKBONES[backbone]()

        # mul_scl_ipt='cat' doubles every lateral channel count.
        channels = [c * 2 for c in LATERAL_CHANNELS[backbone]]
        # cxt_num=3: the three shallower stages are concatenated onto stage 4.
        self.cxt = channels[1:][::-1][-3:]

        self.squeeze_module = nn.Sequential(BasicDecBlk(channels[0] + sum(self.cxt), channels[0]))
        self.decoder = Decoder(channels)

    def forward_enc(self, x):
        x1, x2, x3, x4 = self.bb(x)
        B, C, H, W = x.shape
        x1_, x2_, x3_, x4_ = self.bb(_interp(x, (H // 2, W // 2)))
        x1 = torch.cat([x1, _interp(x1_, x1.shape[2:])], dim=1)
        x2 = torch.cat([x2, _interp(x2_, x2.shape[2:])], dim=1)
        x3 = torch.cat([x3, _interp(x3_, x3.shape[2:])], dim=1)
        x4 = torch.cat([x4, _interp(x4_, x4.shape[2:])], dim=1)
        x4 = torch.cat(
            (
                _interp(x1, x4.shape[2:]),
                _interp(x2, x4.shape[2:]),
                _interp(x3, x4.shape[2:]),
                x4,
            ),
            dim=1,
        )
        return x1, x2, x3, x4

    def forward(self, x):
        x1, x2, x3, x4 = self.forward_enc(x)
        x4 = self.squeeze_module(x4)
        return self.decoder([x, x1, x2, x3, x4])
