# src/models/ssdlite.py
#
# Two build functions:
#   build_ssdlite_vanilla()  ← stock torchvision SSDLite (used for baseline)
#   build_ssdlite()          ← SSDLite + BiFPN neck     (used for hybrid)
#
# ─────────────────────────────────────────────────────────────────────────────
# VANILLA SSDLite (baseline)
# ─────────────────────────────────────────────────────────────────────────────
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.ssdlite import SSDLiteHead
import torch.nn as nn


def build_ssdlite_vanilla(num_classes: int, pretrained: bool = True):
    """
    Stock SSDLite-MobileNetV3-Large from torchvision.
    No neck modifications — pure baseline.

    num_classes : total classes including background
                  e.g. 8 = 7 foreground diseases + 1 background
    """
    if pretrained:
        # weights=DEFAULT internally forces weights_backbone=None,
        # which makes torchvision build the backbone with reduce_tail=True.
        model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
    else:
        # weights_backbone=None is REQUIRED here to force reduce_tail=True.
        # Without it, torchvision defaults weights_backbone=IMAGENET1K_V1 and
        # builds reduce_tail=False (full 960 channels) — which wouldn't match
        # the trainer's architecture (which uses pretrained=True → reduce_tail=True
        # → halved 480 channels) and would break checkpoint loading at eval time.
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
        )

    # Replace the ENTIRE head (not just classification_head sub-module).
    # If we only swap model.head.classification_head with a SSDLiteHead instance,
    # SSD's forward calls self.classification_head(x) and gets back a dict
    # {"cls_logits":..., "bbox_regression":...} instead of a tensor — which makes
    # compute_loss iterate over dict keys (strings), causing 'str has no .size()'.
    num_foreground = num_classes - 1
    # Read per-level in_channels from existing regression head Conv2d layers
    in_channels = [m[-1].in_channels
                   for m in model.head.regression_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head = SSDLiteHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=nn.BatchNorm2d,
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [SSDLite Vanilla] MobileNetV3-Large backbone")
    print(f"  [SSDLite Vanilla] {num_foreground} foreground classes")
    print(f"  [SSDLite Vanilla] Feature levels: {len(in_channels)}  channels: {in_channels}")
    print(f"  [SSDLite Vanilla] Parameters: {total_params:.1f}M")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SSDLite + BiFPN neck (hybrid baseline)
# ─────────────────────────────────────────────────────────────────────────────
# SSDLite with MobileNetV3-Large backbone and BiFPN neck.
#
# Architecture:
#   Backbone : MobileNetV3-Large  →  3 feature maps at strides 8 / 16 / 32
#   Neck     : BiFPN (2 stacked blocks, 256 channels)
#               • Top-down  : P5 → P4_td → P3_out
#               • Bottom-up : P3_out → P4_out → P5_out
#               • Learnable fast-normalized fusion weights per node
#   Head     : SSDLiteHead (depthwise-separable conv, BN)
#   Anchors  : DefaultBoxGenerator  –  6 anchors / location × 3 levels

from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.ssd import SSD, DefaultBoxGenerator
from torchvision.models.detection.ssdlite import SSDLiteHead


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dw_sep_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Depthwise-separable conv → BN → ReLU6."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch,  3, padding=1, groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BiFPN
# ─────────────────────────────────────────────────────────────────────────────

class BiFPNBlock(nn.Module):
    """
    One BiFPN fusion block for 3 feature levels  (P3 / P4 / P5).

    Top-down path  (coarse → fine):
        P4_td  = Conv( w1·P4  + w2·upsample(P5)         )
        P3_out = Conv( w1·P3  + w2·upsample(P4_td)      )

    Bottom-up path (fine → coarse):
        P4_out = Conv( w1·P4  + w2·P4_td + w3·pool(P3_out) )
        P5_out = Conv( w1·P5  + w2·pool(P4_out)             )

    Fusion uses fast-normalised weights  (Tan et al., EfficientDet 2020).
    """

    def __init__(self, channels: int, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon

        # Top-down conv nodes
        self.td_conv_p4 = _dw_sep_conv(channels, channels)
        self.td_conv_p3 = _dw_sep_conv(channels, channels)

        # Bottom-up conv nodes
        self.bu_conv_p4 = _dw_sep_conv(channels, channels)
        self.bu_conv_p5 = _dw_sep_conv(channels, channels)

        # Learnable fusion weights  (initialised to 1 → equal weighting)
        self.w_td_p4 = nn.Parameter(torch.ones(2, dtype=torch.float32))  # P4_in  + upsample(P5_in)
        self.w_td_p3 = nn.Parameter(torch.ones(2, dtype=torch.float32))  # P3_in  + upsample(P4_td)
        self.w_bu_p4 = nn.Parameter(torch.ones(3, dtype=torch.float32))  # P4_in  + P4_td + pool(P3_out)
        self.w_bu_p5 = nn.Parameter(torch.ones(2, dtype=torch.float32))  # P5_in  + pool(P4_out)

    # ------------------------------------------------------------------
    def _fuse(self, features: List[torch.Tensor], w: nn.Parameter) -> torch.Tensor:
        """Fast-normalised weighted sum: w_i / (Σw + ε)."""
        weights = torch.relu(w) + self.epsilon
        weights = weights / weights.sum()
        return sum(wi * f for wi, f in zip(weights, features))

    # ------------------------------------------------------------------
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features

        # ── top-down ──────────────────────────────────────────────────
        p4_td = self.td_conv_p4(
            self._fuse(
                [p4, F.interpolate(p5, size=p4.shape[-2:], mode='nearest')],
                self.w_td_p4,
            )
        )
        p3_out = self.td_conv_p3(
            self._fuse(
                [p3, F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')],
                self.w_td_p3,
            )
        )

        # ── bottom-up ─────────────────────────────────────────────────
        p4_out = self.bu_conv_p4(
            self._fuse(
                [p4, p4_td, F.adaptive_avg_pool2d(p3_out, p4.shape[-2:])],
                self.w_bu_p4,
            )
        )
        p5_out = self.bu_conv_p5(
            self._fuse(
                [p5, F.adaptive_avg_pool2d(p4_out, p5.shape[-2:])],
                self.w_bu_p5,
            )
        )

        return [p3_out, p4_out, p5_out]


class BiFPN(nn.Module):
    """
    Full BiFPN neck:
        1. Lateral 1×1 conv projections  →  uniform `out_channels`
        2. N stacked BiFPN fusion blocks
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_repeats: int = 2,
    ):
        super().__init__()

        # Project each backbone feature map to `out_channels`
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
            )
            for in_ch in in_channels_list
        ])

        # Stack of BiFPN fusion blocks
        self.blocks = nn.ModuleList([
            BiFPNBlock(out_channels) for _ in range(num_repeats)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Lateral projections
        features = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        # Stacked BiFPN fusion
        for block in self.blocks:
            features = block(features)
        return features


# ─────────────────────────────────────────────────────────────────────────────
# Backbone + BiFPN wrapper  (satisfies torchvision SSD backbone contract)
# ─────────────────────────────────────────────────────────────────────────────

class MobileNetV3BiFPN(nn.Module):
    """
    MobileNetV3-Large backbone split into 3 stages, followed by a BiFPN neck.

    Feature extraction points in MobileNetV3-Large:
        stage1  →  features[:7]   : stride  8, 40 ch   (P3)
        stage2  →  features[7:13] : stride 16, 112 ch  (P4)
        stage3  →  features[13:]  : stride 32, 960 ch  (P5)

    Output: OrderedDict {'0': P3_fused, '1': P4_fused, '2': P5_fused}
            all with `out_channels` channels.
    """

    _BACKBONE_CHANNELS = [40, 112, 960]   # channels at strides 8 / 16 / 32

    def __init__(
        self,
        pretrained: bool = True,
        out_channels: int = 256,
        num_repeats: int = 2,
    ):
        super().__init__()

        weights  = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        feats    = backbone.features          # Sequential of 17 layers (0–16)

        self.stage1 = feats[:7]               # → stride  8,  40 ch
        self.stage2 = feats[7:13]             # → stride 16, 112 ch
        self.stage3 = feats[13:]              # → stride 32, 960 ch

        self.bifpn = BiFPN(
            in_channels_list=self._BACKBONE_CHANNELS,
            out_channels=out_channels,
            num_repeats=num_repeats,
        )

        # torchvision SSD reads this attribute to size the default head
        self.out_channels = [out_channels] * 3

    def forward(self, x: torch.Tensor) -> OrderedDict:
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)

        p3, p4, p5 = self.bifpn([p3, p4, p5])

        return OrderedDict([
            ('0', p3),   # stride  8
            ('1', p4),   # stride 16
            ('2', p5),   # stride 32
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ssdlite(num_classes: int, pretrained: bool = True) -> SSD:
    """
    Build SSDLite with MobileNetV3-Large backbone and BiFPN neck.

    Args:
        num_classes : Total classes including background  (10 for this project:
                      0 = background, 1–9 = disease classes)
        pretrained  : Load ImageNet-pretrained MobileNetV3-Large backbone

    Returns:
        torchvision SSD model ready for training / evaluation
    """
    out_channels = 256
    num_repeats  = 2

    backbone = MobileNetV3BiFPN(
        pretrained=pretrained,
        out_channels=out_channels,
        num_repeats=num_repeats,
    )

    # 3 levels × 6 anchors/location
    # aspect_ratios=[2,3] →  1:1, 1:1(extra), 2:1, 1:2, 3:1, 1:3  = 6 per loc
    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[[2, 3], [2, 3], [2, 3]],
        min_ratio=0.15,
        max_ratio=0.90,
    )

    num_anchors = anchor_generator.num_anchors_per_location()   # [6, 6, 6]

    head = SSDLiteHead(
        in_channels=[out_channels] * 3,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=nn.BatchNorm2d,
    )

    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(416, 416),
        num_classes=num_classes,
        head=head,
        # Transforms.py already normalises → disable internal normalisation
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
    )

    if pretrained:
        print("  [SSDLite+BiFPN] MobileNetV3-Large backbone: ImageNet pretrained")
    else:
        print("  [SSDLite+BiFPN] MobileNetV3-Large backbone: random init")

    print(
        f"  [SSDLite+BiFPN] Neck: BiFPN  "
        f"({num_repeats} blocks → 256 ch)  "
        f"| Levels: P3/P4/P5  | Anchors: 6/loc"
    )

    return model
