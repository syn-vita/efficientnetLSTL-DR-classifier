from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_baseline_model(num_classes: int, use_pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
    try:
        model = efficientnet_b0(weights=weights)
    except Exception as exc:
        print(f"Warning: failed to load pretrained weights ({exc}). Falling back to random init.")
        model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
