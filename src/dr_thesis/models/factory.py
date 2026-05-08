from __future__ import annotations

import torch
from torch import nn

from .baseline import build_baseline_model
from .cbam import EfficientNetB0WithCBAM
from .lstl import EfficientNetB0WithLSTL


def build_model(
    model_key: str,
    num_classes: int,
    device: torch.device | str | None = None,
    use_pretrained: bool = False,
) -> nn.Module:
    if model_key == "baseline":
        model = build_baseline_model(num_classes=num_classes, use_pretrained=use_pretrained)
    elif model_key == "cbam":
        model = EfficientNetB0WithCBAM(num_classes=num_classes, use_pretrained=use_pretrained)
    elif model_key == "lstl":
        model = EfficientNetB0WithLSTL(num_classes=num_classes, use_pretrained=use_pretrained)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    if device is not None:
        model = model.to(device)
    return model
