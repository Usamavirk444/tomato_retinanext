"""
RetinaNet — ConvNeXt-Tiny + BiFPN×3  (NO CBAM)
================================================
Ablation variant sitting between:
    ConvNeXt + FPN        (retinanet_convnext_baseline)   — done ✓
    ConvNeXt + BiFPN      (THIS file)                      — ablation step
    ConvNeXt + BiFPN+CBAM (retinanet_hybrid v3)           — done ✓

Isolates the contribution of BiFPN alone, without CBAM attention.
Everything else (backbone, head, anchors, hyperparameters) is identical
to retinanet_hybrid so the comparison is fair.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

# Reuse the existing BiFPN and ConvNeXt backbone from src/
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.hybrid.bifpn import BiFPN
from src.models.hybrid.convnext_backbone import ConvNeXtTinyBody, build_convnext_backbone


# ──────────────────────────────────────────────────────────────────────────────
# Backbone with BiFPN neck — NO CBAM
# ──────────────────────────────────────────────────────────────────────────────
class BackboneWithBiFPN(nn.Module):
    """
    ConvNeXt-Tiny body + BiFPN (3 repeats), no CBAM.
    Exposes the same interface RetinaNet expects:
        forward(x) -> OrderedDict of multi-scale features
        .out_channels (int)
    """

    def __init__(self, body: ConvNeXtTinyBody,
                 num_bifpn_repeats: int = 3,
                 out_channels: int = 256):
        super().__init__()
        self.body = body
        self.out_channels = out_channels

        extra_blocks = LastLevelP6P7(
            in_channels=out_channels, out_channels=out_channels)

        self.fpn = BiFPN(
            in_channels_list=ConvNeXtTinyBody.OUT_CHANNELS,  # [192, 384, 768]
            out_channels=out_channels,
            num_repeats=num_bifpn_repeats,
            extra_blocks=extra_blocks,
        )

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        feats = self.body(x)       # {"0": C3, "1": C4, "2": C5}
        return self.fpn(feats)     # {"0": P3 ... "p7": P7}


# ──────────────────────────────────────────────────────────────────────────────
# Anchor generator — identical to retinanet_hybrid
# ──────────────────────────────────────────────────────────────────────────────
def _default_anchor_generator() -> AnchorGenerator:
    anchor_sizes = (
        (32, 40, 50),
        (64, 80, 101),
        (128, 161, 203),
        (256, 322, 406),
        (512, 645, 812),
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


# ──────────────────────────────────────────────────────────────────────────────
# Public builder
# ──────────────────────────────────────────────────────────────────────────────
def build_retinanet_convnext_bifpn(num_classes: int,
                                    pretrained: bool = True,
                                    num_bifpn_repeats: int = 3):
    """
    ConvNeXt-Tiny + BiFPN×3 + RetinaNet head.  No CBAM.

    Args:
        num_classes:  includes background (10 = 9 fg + 1 bg).
        pretrained:   load ImageNet weights for ConvNeXt-T body.
    """
    num_foreground = num_classes - 1

    body = build_convnext_backbone(pretrained=pretrained)
    backbone = BackboneWithBiFPN(
        body=body,
        num_bifpn_repeats=num_bifpn_repeats,
        out_channels=256,
    )

    anchor_generator = _default_anchor_generator()

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_foreground,
        anchor_generator=anchor_generator,
    )

    total_params  = sum(p.numel() for p in model.parameters()) / 1e6
    body_params   = sum(p.numel() for p in body.parameters()) / 1e6
    bifpn_params  = sum(p.numel() for p in backbone.fpn.parameters()) / 1e6

    print(f"  [BiFPN-only] Backbone : ConvNeXt-Tiny")
    print(f"  [BiFPN-only] Neck     : BiFPN×{num_bifpn_repeats} (256 ch)  — NO CBAM")
    print(f"  [BiFPN-only] Body  params : {body_params:.2f} M")
    print(f"  [BiFPN-only] BiFPN params : {bifpn_params:.2f} M")
    print(f"  [BiFPN-only] TOTAL params : {total_params:.2f} M")

    return model
