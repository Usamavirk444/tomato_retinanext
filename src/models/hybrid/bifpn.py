# src/models/hybrid/bifpn.py
"""
BiFPN — Bidirectional Feature Pyramid Network
─────────────────────────────────────────────
Tan et al., "EfficientDet: Scalable and Efficient Object Detection",
CVPR 2020. https://arxiv.org/abs/1911.09070

This BiFPN is a drop-in replacement for the standard top-down-only
FeaturePyramidNetwork (FPN) used in `retinanet_resnet50_fpn_v2`.

Standard FPN (one-way):
        C5 ── 1×1 ──► P5
                       │ upsample
        C4 ── 1×1 ──► P4
                       │ upsample
        C3 ── 1×1 ──► P3

BiFPN (bidirectional + weighted, repeated `num_repeats` times):
        Top-down  : P5  →  P4_td  →  P3_out
        Bottom-up : P3_out  →  P4_out  →  P5_out
    + learnable fast-normalised fusion weights at every node.

Why this matters for tomato leaf disease:
    Lesion sizes vary wildly across our 9 classes — early-blight
    specks are tiny (P3 territory) while mosaic-virus / yellow-leaf-
    curl discoloration covers most of the leaf (P5 territory).
    Bidirectional, weighted fusion lets the network combine fine
    spatial detail from P3 with semantic context from P5 in BOTH
    directions, instead of only top-down (FPN) or bottom-up (PAN).

Note on input shape:
    RetinaNet's `BackboneWithFPN` body returns three feature maps
    from ResNet-50 stages C3/C4/C5 with channels [512, 1024, 2048].
    P6/P7 are added AFTER this module by torchvision's `LastLevelP6P7`
    extra-blocks layer (kept unchanged so anchor matching is identical
    to the baseline).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _depthwise_separable_conv(channels: int) -> nn.Sequential:
    """
    Depthwise-separable 3×3 conv used by EfficientDet's BiFPN nodes.
    Lightweight refinement after each weighted fusion.
    """
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                  groups=channels, bias=False),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.SiLU(inplace=True),
    )


# ────────────────────────────────────────────────────────────────────
# One BiFPN fusion block (3 levels: P3, P4, P5)
# ────────────────────────────────────────────────────────────────────
class BiFPNBlock(nn.Module):
    """
    A single BiFPN fusion block over three feature levels.

        Top-down path  (semantic ↓ to spatial):
            P4_td  = Conv( w·P4_in  + w·upsample(P5_in)        )
            P3_out = Conv( w·P3_in  + w·upsample(P4_td)        )

        Bottom-up path (spatial ↑ to semantic):
            P4_out = Conv( w·P4_in  + w·P4_td + w·pool(P3_out) )
            P5_out = Conv( w·P5_in  + w·pool(P4_out)           )

    Weights are positive (ReLU) and fast-normalised (Eq. 4 of paper).
    """

    def __init__(self, channels: int, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon

        # Top-down conv refinements.
        self.td_conv_p4 = _depthwise_separable_conv(channels)
        self.td_conv_p3 = _depthwise_separable_conv(channels)

        # Bottom-up conv refinements.
        self.bu_conv_p4 = _depthwise_separable_conv(channels)
        self.bu_conv_p5 = _depthwise_separable_conv(channels)

        # Per-edge learnable fusion weights (init = 1 → equal weighting).
        self.w_td_p4 = nn.Parameter(torch.ones(2))   # [P4_in, up(P5_in)]
        self.w_td_p3 = nn.Parameter(torch.ones(2))   # [P3_in, up(P4_td)]
        self.w_bu_p4 = nn.Parameter(torch.ones(3))   # [P4_in, P4_td, pool(P3_out)]
        self.w_bu_p5 = nn.Parameter(torch.ones(2))   # [P5_in, pool(P4_out)]

    def _fuse(self, features: List[torch.Tensor], w: nn.Parameter) -> torch.Tensor:
        """Fast-normalised weighted sum (paper Eq. 4)."""
        weights = F.relu(w) + self.epsilon
        weights = weights / weights.sum()
        return sum(weights[i] * f for i, f in enumerate(features))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features

        # Top-down.
        p4_td = self.td_conv_p4(self._fuse(
            [p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")],
            self.w_td_p4,
        ))
        p3_out = self.td_conv_p3(self._fuse(
            [p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")],
            self.w_td_p3,
        ))

        # Bottom-up.
        p4_out = self.bu_conv_p4(self._fuse(
            [p4, p4_td, F.adaptive_avg_pool2d(p3_out, p4.shape[-2:])],
            self.w_bu_p4,
        ))
        p5_out = self.bu_conv_p5(self._fuse(
            [p5, F.adaptive_avg_pool2d(p4_out, p5.shape[-2:])],
            self.w_bu_p5,
        ))

        return [p3_out, p4_out, p5_out]


# ────────────────────────────────────────────────────────────────────
# Full BiFPN neck (drop-in replacement for torchvision FPN)
# ────────────────────────────────────────────────────────────────────
class BiFPN(nn.Module):
    """
    Drop-in replacement for `torchvision.ops.FeaturePyramidNetwork`
    used inside `BackboneWithFPN`.

    Forward contract (matches torchvision FPN):
        Input  : OrderedDict of feature maps from the backbone body
                 — keys '0','1','2' (C3/C4/C5 in our case),
                   values shape (B, C_i, H_i, W_i).
        Output : OrderedDict with the SAME keys, all with `out_channels`
                 channels, post-BiFPN. If `extra_blocks` is given (e.g.
                 LastLevelP6P7), it is applied at the end exactly as
                 torchvision FPN does, producing extra entries.

    Args:
        in_channels_list : channels at each input level (e.g. [512,1024,2048]).
        out_channels     : uniform channel count after lateral projection
                           (RetinaNet uses 256 — keep this for parity).
        num_repeats      : number of stacked BiFPN blocks
                           (EfficientDet-D0 uses 3; we use 2 to match
                           the parameter budget of the baseline FPN).
        extra_blocks     : optional torchvision ExtraFPNBlock for P6/P7.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_repeats: int = 2,
        extra_blocks: nn.Module | None = None,
    ):
        super().__init__()
        assert len(in_channels_list) == 3, (
            "This BiFPN expects exactly 3 backbone levels (C3/C4/C5)."
        )

        # Lateral 1×1 projections to a uniform channel count.
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
            for in_ch in in_channels_list
        ])

        self.blocks = nn.ModuleList([
            BiFPNBlock(out_channels) for _ in range(num_repeats)
        ])

        # Hold the same `extra_blocks` instance the original FPN had,
        # so P6/P7 generation behaves identically to the baseline.
        self.extra_blocks = extra_blocks

        self.out_channels = out_channels

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        names = list(x.keys())          # e.g. ['0', '1', '2']
        body_feats = list(x.values())   # C3, C4, C5  (pre-lateral, original channels)

        # 1. Lateral projection to common channels.
        feats = [conv(f) for conv, f in zip(self.lateral_convs, body_feats)]

        # 2. Stacked BiFPN blocks (bidirectional + weighted fusion).
        for block in self.blocks:
            feats = block(feats)

        # 3. Optional extra blocks (LastLevelP6P7) — produce P6, P7.
        #
        #    torchvision's ExtraFPNBlock signature mirrors what the original
        #    FPN passes:
        #        extra_blocks(results=p, x=c, names=names)
        #          • p = post-pyramid features (uniform out_channels)
        #          • c = pre-lateral backbone features (varied channels)
        #    LastLevelP6P7 picks p[-1] vs c[-1] based on `use_P5`. For
        #    RetinaNet-v2, use_P5=True (in=out=256), so c is unused — but
        #    we still pass body_feats here to match the torchvision contract
        #    exactly, in case anyone swaps in a different ExtraFPNBlock.
        if self.extra_blocks is not None:
            feats, names = self.extra_blocks(feats, body_feats, names)

        return OrderedDict([(n, f) for n, f in zip(names, feats)])
