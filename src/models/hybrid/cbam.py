# src/models/hybrid/cbam.py
"""
CBAM — Convolutional Block Attention Module
─────────────────────────────────────────────
Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
https://arxiv.org/abs/1807.06521

Two sequential sub-modules:

    Channel Attention :  global avg-pool + global max-pool → shared MLP
                         → sigmoid → channel-wise gate
    Spatial Attention :  channel-wise mean + max → 7×7 conv
                         → sigmoid → spatial gate

In the hybrid model, one CBAM is applied to each pyramid output
(P3, P4, P5, P6, P7) of the BiFPN, BEFORE features go to the
classification / regression heads.

Justification for our paper:
    All three baselines (SSDLite, YOLOv8n, RetinaNet) failed on the
    "Healthy" class — they confuse healthy leaf texture with disease
    symptoms. CBAM lets the network learn (a) which channels carry
    disease-relevant information and (b) which spatial regions of the
    leaf actually contain a lesion, suppressing healthy-leaf texture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention — squeezes spatial dims with avg + max pool,
    then a shared 2-layer MLP produces per-channel gate weights.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum the two pooled paths through the SAME MLP (CBAM Eq. 2).
        gate = self.sigmoid(
            self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        )
        return x * gate


class SpatialAttention(nn.Module):
    """
    Spatial attention — channel-wise mean + max → 7×7 conv → sigmoid.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "CBAM uses 7×7 conv (or 3×3 for small maps)"
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling.
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        gate = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class CBAM(nn.Module):
    """
    Full CBAM block: channel attention → spatial attention.

    Args:
        channels    : input/output feature map channels (unchanged).
        reduction   : channel-MLP reduction ratio (paper default = 16).
        kernel_size : spatial-attention conv kernel (paper default = 7).
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
