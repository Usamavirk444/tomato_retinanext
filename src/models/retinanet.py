# src/models/retinanet.py

from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def build_retinanet(num_classes: int, pretrained: bool = True,
                    box_loss: str = "giou"):
    """
    Build RetinaNet with ResNet50-FPN backbone.
    Single-stage detector with Focal Loss.
    num_classes includes background (e.g. 10 = 9 foreground + 1 background)
    RetinaNet internally uses num_classes - 1 foreground classes.

    box_loss: regression loss for the box subnet.

        "giou"  torchvision's default FOR THIS ARCHITECTURE. The factory
                retinanet_resnet50_fpn_v2() overrides the head's default
                with `head.regression_head._loss_type = "giou"`. This is
                what produced every published S1 result, so it stays the
                default here and those results remain reproducible.

        "l1"    the RetinaNetRegressionHead class default, and what every
                ConvNeXt-based stage (S2-S5) uses, because they are built
                by calling RetinaNet(...) directly rather than through the
                factory. Pass "l1" to remove that confound and make the
                S1 -> S2 step differ only in the backbone.

    Torchvision supports l1, smooth_l1, ciou, diou and giou; only the two
    relevant here are allowed, so a typo fails loudly instead of silently
    training the wrong model for several hours.
    """
    if box_loss not in ("giou", "l1"):
        raise ValueError(f"box_loss must be 'giou' or 'l1', got {box_loss!r}")

    if pretrained:
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        )
    else:
        model = retinanet_resnet50_fpn_v2(weights=None)

    num_foreground = num_classes - 1
    num_anchors    = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_foreground,
    )

    # Replacing the classification head above leaves the regression head
    # untouched, so whatever the factory set is still in place. Override
    # only when asked.
    model.head.regression_head._loss_type = box_loss

    actual = model.head.regression_head._loss_type
    if actual != box_loss:
        raise RuntimeError(
            f"box regression loss is {actual!r} after construction, "
            f"expected {box_loss!r}")

    print(f"  RetinaNet: {num_foreground} foreground classes (labels 0-{num_foreground-1})")
    print(f"  RetinaNet: box regression loss = {actual}")

    return model
