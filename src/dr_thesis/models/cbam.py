from __future__ import annotations

import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(torch.nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class EfficientNetB0WithCBAM(torch.nn.Module):
    def __init__(self, num_classes: int, use_pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        try:
            self.base = efficientnet_b0(weights=weights)
        except Exception as exc:
            print(f"Warning: failed to load pretrained weights ({exc}). Falling back to random init.")
            self.base = efficientnet_b0(weights=None)
        in_features = self.base.classifier[1].in_features
        self.cbam = CBAM(in_features)
        self.base.classifier[1] = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.features(x)
        x = self.cbam(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.classifier(x)
        return x
