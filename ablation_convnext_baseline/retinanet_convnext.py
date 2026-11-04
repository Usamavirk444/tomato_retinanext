"""
RetinaNet-ConvNeXt-T ablation baseline — standalone model builder.

Same ConvNeXt-Tiny backbone as the hybrid, but:
  - Standard top-down FPN (NOT BiFPN)
  - No CBAM attention
  - Default RetinaNet detection head + anchors

Purpose: isolate "how much of the hybrid's gain comes from the backbone vs
from BiFPN+CBAM." With this baseline, the paper can say:
    "ConvNeXt-T backbone alone adds +X pp over ResNet-50.
     BiFPN+CBAM contribute the additional +Y pp on top."

DOES NOT modify any existing training code. Only imports from src/.
Reuses the same ConvNeXt body class as the hybrid (read-only import).
"""

from __future__ import annotations
import os
import sys
import ssl
import urllib.request

# SSL bypass for university MITM proxy (only fires on first weight download)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    _ctx = ssl._create_unverified_context()
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
    )
except Exception:
    pass

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork, LastLevelP6P7)

# Find project root and import the ConvNeXt body (read-only)
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.hybrid.convnext_backbone import (
    ConvNeXtTinyBody, build_convnext_backbone)


class BackboneWithStandardFPN(nn.Module):
    """
    ConvNeXt-Tiny body + standard FeaturePyramidNetwork + LastLevelP6P7.

    Forward contract matches torchvision's BackboneWithFPN:
        forward(x) -> OrderedDict of multi-scale features  ("0".."p7")
        .out_channels (int)
    """
    def __init__(self, body: ConvNeXtTinyBody, out_channels: int = 256):
        super().__init__()
        self.body = body
        self.out_channels = out_channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=ConvNeXtTinyBody.OUT_CHANNELS,  # [192, 384, 768]
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        feats = self.body(x)
        return self.fpn(feats)


def _default_anchor_generator() -> AnchorGenerator:
    """Same as torchvision retinanet_resnet50_fpn_v2 anchors (32-812 px)."""
    anchor_sizes = (
        (32, 40, 50),
        (64, 80, 101),
        (128, 161, 203),
        (256, 322, 406),
        (512, 645, 812),
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


def build_retinanet_convnext(num_classes: int, pretrained: bool = True):
    """
    Build RetinaNet with ConvNeXt-T backbone + standard FPN + default head.

    Args:
        num_classes:  includes background (10 = 9 fg + 1 bg)
        pretrained:   load ImageNet weights for ConvNeXt-T body

    Returns: torchvision RetinaNet model.
    """
    num_foreground = num_classes - 1

    body = build_convnext_backbone(pretrained=pretrained)
    backbone = BackboneWithStandardFPN(body=body, out_channels=256)
    anchor_generator = _default_anchor_generator()

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_foreground,
        anchor_generator=anchor_generator,
    )

    # Logging
    total = sum(p.numel() for p in model.parameters()) / 1e6
    body_p = sum(p.numel() for p in body.parameters()) / 1e6
    fpn_p = sum(p.numel() for p in backbone.fpn.parameters()) / 1e6
    head_p = sum(p.numel() for p in model.head.parameters()) / 1e6
    print(f"  [RetinaNet-ConvNeXt-T baseline]")
    print(f"  Backbone : ConvNeXt-Tiny  (out_channels={ConvNeXtTinyBody.OUT_CHANNELS})")
    print(f"  Neck     : Standard FPN (256 ch) + LastLevelP6P7")
    print(f"  Head     : Standard RetinaNet (focal loss)")
    print(f"  Anchors  : default (32-512 px), 9 per location")
    print(f"  Body params: {body_p:.2f} M")
    print(f"  FPN  params: {fpn_p:.2f} M")
    print(f"  Head params: {head_p:.2f} M")
    print(f"  TOTAL params: {total:.2f} M")

    return model
