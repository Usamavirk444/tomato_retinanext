"""
RetinaNet — ConvNeXt-Tiny + standard FPN + CBAM  (NO BiFPN)
============================================================
Missing ablation cell requested by reviewer (Critical):

    S1  ResNet-50 + FPN                      → done ✓
    S2  ConvNeXt-T + standard FPN             → done ✓   (retinanet_convnext_baseline)
    S2.5 ConvNeXt-T + standard FPN + CBAM    → THIS FILE  ← the missing cell
    S3  ConvNeXt-T + BiFPN (no CBAM)         → done ✓   (ablation_bifpn_only)
    S4  ConvNeXt-T + BiFPN + CBAM            → done ✓   (retinanet_hybrid — RetinaNeXt)

Purpose:
    Isolates CBAM's contribution independently of BiFPN.
    Comparing S2 vs S2.5 shows "CBAM alone" (no BiFPN).
    Comparing S3 vs S4  shows "CBAM on top of BiFPN".
    If (S4 - S3) > (S2.5 - S2), BiFPN+CBAM truly synergise.
    If not, CBAM's gain is independent of the neck type.

Architecture:
    backbone  : ConvNeXt-Tiny body (ImageNet pretrained, frozen/unfrozen like others)
    neck      : standard top-down FeaturePyramidNetwork + LastLevelP6P7  (256 ch)
    attention : CBAM applied to each of the 5 FPN output levels (P3..P7)
    head      : default RetinaNet classification + regression head (focal loss)
    anchors   : identical to all other variants (32-512 px, 9/loc)

Nothing in any other folder is imported or modified.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork, LastLevelP6P7)

import sys
import os

HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Read-only imports from src/ — nothing is modified
from src.models.hybrid.cbam             import CBAM
from src.models.hybrid.convnext_backbone import ConvNeXtTinyBody, build_convnext_backbone


# ──────────────────────────────────────────────────────────────────────────────
# Backbone: ConvNeXt body  +  standard FPN  +  CBAM per level
# ──────────────────────────────────────────────────────────────────────────────
class BackboneWithFPNCBAM(nn.Module):
    """
    ConvNeXt-Tiny body → standard FeaturePyramidNetwork (+ LastLevelP6P7)
    → CBAM attention on each of the 5 pyramid levels (P3, P4, P5, P6, P7).

    Exposes the same interface RetinaNet expects:
        forward(x) -> OrderedDict of multi-scale feature maps
        .out_channels (int)
    """

    NUM_LEVELS = 5   # P3, P4, P5, P6, P7

    def __init__(self, body: ConvNeXtTinyBody, out_channels: int = 256):
        super().__init__()
        self.body        = body
        self.out_channels = out_channels

        # Standard top-down FPN with P6 and P7 extra levels — identical
        # to what retinanet_convnext_baseline uses.
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=ConvNeXtTinyBody.OUT_CHANNELS,  # [192, 384, 768]
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels, out_channels),
        )

        # One CBAM block per FPN output level (P3..P7)
        self.cbams = nn.ModuleList([
            CBAM(channels=out_channels, reduction=16, kernel_size=7)
            for _ in range(self.NUM_LEVELS)
        ])

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        # 1. Extract multi-scale body features: {0: C3, 1: C4, 2: C5}
        body_feats = self.body(x)

        # 2. Top-down FPN fusion:  {0: P3, 1: P4, 2: P5, p6: P6, p7: P7}
        pyramid    = self.fpn(body_feats)

        # 3. Apply CBAM to each level
        keys      = list(pyramid.keys())
        attended  = OrderedDict()
        for i, k in enumerate(keys):
            cbam_idx      = min(i, len(self.cbams) - 1)   # safety clamp
            attended[k]   = self.cbams[cbam_idx](pyramid[k])

        return attended


# ──────────────────────────────────────────────────────────────────────────────
# Anchor generator — identical across ALL ablation variants
# ──────────────────────────────────────────────────────────────────────────────
def _default_anchor_generator() -> AnchorGenerator:
    """Same as torchvision retinanet_resnet50_fpn_v2 (32-812 px, 9/loc)."""
    anchor_sizes = (
        (32,  40,  50),
        (64,  80, 101),
        (128, 161, 203),
        (256, 322, 406),
        (512, 645, 812),
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


# ──────────────────────────────────────────────────────────────────────────────
# Public builder
# ──────────────────────────────────────────────────────────────────────────────
def build_retinanet_convnext_fpn_cbam(num_classes: int,
                                       pretrained: bool = True):
    """
    Build RetinaNet with ConvNeXt-Tiny + standard FPN + CBAM.  No BiFPN.

    Args:
        num_classes : total classes including background (10 = 9 fg + 1 bg).
        pretrained  : load ImageNet weights for the ConvNeXt-Tiny body.

    Returns:
        torchvision.models.detection.RetinaNet (ready to .to(device))
    """
    num_foreground = num_classes - 1

    body     = build_convnext_backbone(pretrained=pretrained)
    backbone = BackboneWithFPNCBAM(body=body, out_channels=256)

    anchor_generator = _default_anchor_generator()

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_foreground,
        anchor_generator=anchor_generator,
    )

    # ── Parameter accounting ──────────────────────────────────────────────────
    total_p = sum(p.numel() for p in model.parameters())       / 1e6
    body_p  = sum(p.numel() for p in body.parameters())        / 1e6
    fpn_p   = sum(p.numel() for p in backbone.fpn.parameters()) / 1e6
    cbam_p  = sum(p.numel() for p in backbone.cbams.parameters()) / 1e6
    head_p  = sum(p.numel() for p in model.head.parameters())  / 1e6

    print(f"  [FPN+CBAM ablation]")
    print(f"  Backbone  : ConvNeXt-Tiny  {ConvNeXtTinyBody.OUT_CHANNELS}")
    print(f"  Neck      : Standard FPN (256 ch) + LastLevelP6P7  — NO BiFPN")
    print(f"  Attention : CBAM × {BackboneWithFPNCBAM.NUM_LEVELS} levels (P3..P7)")
    print(f"  Anchors   : default (32-812 px), 9 per location")
    print(f"  Foreground classes: {num_foreground}")
    print(f"  Body  params : {body_p:.2f} M")
    print(f"  FPN   params : {fpn_p:.2f} M")
    print(f"  CBAM  params : {cbam_p:.3f} M")
    print(f"  Head  params : {head_p:.2f} M")
    print(f"  TOTAL params : {total_p:.2f} M")

    return model
