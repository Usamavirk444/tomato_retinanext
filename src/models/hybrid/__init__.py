"""
RetinaNet-Hybrid v3 package.

Public API mirrors v1 so the trainer call site does not change:
    from src.models.hybrid import build_retinanet_hybrid

v3 changes vs v1:
  - Backbone: ResNet-50 -> ConvNeXt-Tiny (better fine-grained features)
  - BiFPN repeats: 2 -> 3
  - CBAM: kept (works well in v1)
  - Anchors: kept default (verified by dataset analysis — boxes are large)
"""

from .convnext_backbone import ConvNeXtTinyBody, build_convnext_backbone
from .bifpn import BiFPN, BiFPNBlock
from .cbam import CBAM, ChannelAttention, SpatialAttention
from .retinanet_hybrid import (
    build_retinanet_hybrid,
    count_params,
    BackboneWithBiFPNCBAM,
    CBAMNeck,
)

__all__ = [
    "build_retinanet_hybrid",
    "count_params",
    "ConvNeXtTinyBody",
    "build_convnext_backbone",
    "BackboneWithBiFPNCBAM",
    "CBAMNeck",
    "BiFPN",
    "BiFPNBlock",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
]
