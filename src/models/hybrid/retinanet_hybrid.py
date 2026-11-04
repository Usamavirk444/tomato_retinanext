"""
RetinaNet-Hybrid v3
═══════════════════
ConvNeXt-Tiny backbone  +  BiFPN (3 repeats)  +  CBAM  +  default RetinaNet head.

Why each piece:
  • ConvNeXt-Tiny: fine-grained feature extraction. Tomato disease classes are
    visually similar (dark spots: Early Blight / Late Blight / Septoria) — the
    7×7 depthwise convolutions of ConvNeXt preserve local texture better than
    ResNet's bottleneck blocks.  ImageNet-pretrained, ~28M params.
  • BiFPN×3 + CBAM: kept from v1 (proven to give +1.7pp at seed 123). The
    bidirectional weighted fusion + per-level attention helps multi-scale
    reasoning across our box size distribution (median 210×210, range
    50–640 px).
  • Default RetinaNet anchor generator and head: dataset-analysis showed
    77% of boxes are LARGE (>96px), so the default anchor scheme (32–512)
    matches well. Don't repeat the v2 mistake of shrinking anchors.

Param budget vs baselines (rough):
    Baseline RetinaNet-ResNet50:                ~32 M
    Hybrid v1 (ResNet50 + BiFPN×2 + CBAM):     ~33 M
    Hybrid v3 (ConvNeXt-T + BiFPN×3 + CBAM):   ~37 M

For fairness, the paper compares Hybrid-v3 to a RetinaNet-ConvNeXt-T baseline
with the same backbone but the original FPN (no BiFPN, no CBAM). That isolates
the architectural contribution.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn

from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from .bifpn import BiFPN
from .cbam import CBAM
from .convnext_backbone import ConvNeXtTinyBody, build_convnext_backbone


# ────────────────────────────────────────────────────────────────────
# CBAM-decorated neck — same wrapper as v1
# ────────────────────────────────────────────────────────────────────
class CBAMNeck(nn.Module):
    """
    Wraps the BiFPN neck and applies a CBAM attention block to each
    of its output feature maps (P3..P7).

    Forward contract is identical to the wrapped neck — both input
    and output are OrderedDicts with the same keys.
    """

    def __init__(self, neck: nn.Module, num_levels: int, channels: int = 256):
        super().__init__()
        self.neck = neck
        self.cbams = nn.ModuleList([
            CBAM(channels=channels, reduction=16, kernel_size=7)
            for _ in range(num_levels)
        ])
        self.out_channels = channels

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = self.neck(x)
        keys = list(feats.keys())
        attended = OrderedDict()
        for i, k in enumerate(keys):
            cbam_idx = min(i, len(self.cbams) - 1)
            attended[k] = self.cbams[cbam_idx](feats[k])
        return attended


# ────────────────────────────────────────────────────────────────────
# Backbone with neck — emulates torchvision's BackboneWithFPN
# ────────────────────────────────────────────────────────────────────
class BackboneWithBiFPNCBAM(nn.Module):
    """
    Combines ConvNeXt-Tiny body + BiFPN (3 repeats) + CBAM into a single
    nn.Module that exposes the exact interface RetinaNet expects:
        forward(x) -> OrderedDict of multi-scale features
        .out_channels (int)

    This mirrors what `torchvision.models.detection.backbone_utils.
    BackboneWithFPN` does for ResNet+FPN.
    """

    def __init__(self, body: ConvNeXtTinyBody, num_bifpn_repeats: int = 3,
                 out_channels: int = 256):
        super().__init__()
        self.body = body
        self.out_channels = out_channels

        # P6, P7 are produced from the top of the BiFPN output (not the body)
        # because BiFPN normalises channels to 256, matching what
        # LastLevelP6P7 expects when use_P5=True.
        extra_blocks = LastLevelP6P7(in_channels=out_channels,
                                     out_channels=out_channels)

        bifpn = BiFPN(
            in_channels_list=ConvNeXtTinyBody.OUT_CHANNELS,  # [192, 384, 768]
            out_channels=out_channels,
            num_repeats=num_bifpn_repeats,
            extra_blocks=extra_blocks,
        )
        self.fpn = CBAMNeck(neck=bifpn, num_levels=5, channels=out_channels)

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        feats = self.body(x)        # OrderedDict {"0": C3, "1": C4, "2": C5}
        return self.fpn(feats)      # OrderedDict {"0": P3 ... "p7": P7}


# ────────────────────────────────────────────────────────────────────
# Default RetinaNet anchor generator (5 levels: P3..P7)
# ────────────────────────────────────────────────────────────────────
def _default_anchor_generator() -> AnchorGenerator:
    """
    Same anchors as torchvision's retinanet_resnet50_fpn_v2.
    Dataset analysis confirmed default sizes (32-512) match our box
    distribution (median 210×210). Do NOT change these.
    """
    anchor_sizes = (
        (32, 40, 50),
        (64, 80, 101),
        (128, 161, 203),
        (256, 322, 406),
        (512, 645, 812),
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


# ────────────────────────────────────────────────────────────────────
# Public builder — same signature as v1's build_retinanet_hybrid
# ────────────────────────────────────────────────────────────────────
def build_retinanet_hybrid(num_classes: int, pretrained: bool = True,
                           num_bifpn_repeats: int = 3):
    """
    Build the v3 hybrid: ConvNeXt-T + BiFPN×3 + CBAM + default RetinaNet head.

    Args:
        num_classes:       includes background (e.g. 10 = 9 fg + 1 bg).
                           RetinaNet's sigmoid focal loss internally uses
                           num_classes-1 foreground labels.
        pretrained:        load ImageNet weights for ConvNeXt-T body. Heads
                           are randomly initialized (will train from scratch).
        num_bifpn_repeats: how many stacked BiFPN blocks (default 3).

    Returns:
        torchvision.models.detection.RetinaNet model.
    """
    num_foreground = num_classes - 1

    # 1. Build ConvNeXt-T body and wrap with BiFPN+CBAM neck
    body = build_convnext_backbone(pretrained=pretrained)
    backbone = BackboneWithBiFPNCBAM(
        body=body,
        num_bifpn_repeats=num_bifpn_repeats,
        out_channels=256,
    )

    # 2. Standard RetinaNet anchor generator (5 levels: P3..P7, 9 anchors/loc)
    anchor_generator = _default_anchor_generator()

    # 3. Build the full RetinaNet model. We do NOT load COCO weights for the
    #    heads (those were tied to ResNet-50 features). The heads will train
    #    from scratch — fine for a 22k-image dataset.
    model = RetinaNet(
        backbone=backbone,
        num_classes=num_foreground,
        anchor_generator=anchor_generator,
        # All other defaults match torchvision's retinanet_resnet50_fpn_v2:
        # min_size=800, max_size=1333, image_mean/std=ImageNet,
        # detections_per_img=300, focal alpha=0.25 gamma=2.0,
        # nms_thresh=0.5, score_thresh=0.05, etc.
    )

    # 4. Logging
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    body_params = sum(p.numel() for p in body.parameters()) / 1e6
    bifpn_params = sum(p.numel() for p in backbone.fpn.neck.parameters()) / 1e6
    cbam_params = sum(p.numel() for p in backbone.fpn.cbams.parameters()) / 1e6
    print(f"  [Hybrid-v3] Backbone : ConvNeXt-Tiny  (out_channels={ConvNeXtTinyBody.OUT_CHANNELS})")
    print(f"  [Hybrid-v3] Neck     : BiFPN×{num_bifpn_repeats} (256 ch) + CBAM × 5 levels")
    print(f"  [Hybrid-v3] Anchors  : default (32-512 px), 9 per location")
    print(f"  [Hybrid-v3] Foreground classes: {num_foreground} (labels 0–{num_foreground-1})")
    print(f"  [Hybrid-v3] Body  params: {body_params:.2f} M")
    print(f"  [Hybrid-v3] BiFPN params: {bifpn_params:.2f} M")
    print(f"  [Hybrid-v3] CBAM  params: {cbam_params:.3f} M")
    print(f"  [Hybrid-v3] TOTAL params: {total_params:.2f} M")

    return model


# ────────────────────────────────────────────────────────────────────
# Param accounting helper (handy for the paper)
# ────────────────────────────────────────────────────────────────────
def count_params(model: nn.Module) -> dict:
    """Return total / trainable / per-component param counts in millions."""
    def m(p): return round(p / 1e6, 3)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    body_params = 0
    bifpn_params = 0
    cbam_params = 0
    head_params = 0
    for name, p in model.named_parameters():
        if "backbone.body" in name:
            body_params += p.numel()
        elif "backbone.fpn.neck" in name:
            bifpn_params += p.numel()
        elif "backbone.fpn.cbams" in name:
            cbam_params += p.numel()
        elif "head" in name:
            head_params += p.numel()

    return {
        "total_M":     m(total),
        "trainable_M": m(trainable),
        "body_M":      m(body_params),
        "bifpn_M":     m(bifpn_params),
        "cbam_M":      m(cbam_params),
        "head_M":      m(head_params),
    }
