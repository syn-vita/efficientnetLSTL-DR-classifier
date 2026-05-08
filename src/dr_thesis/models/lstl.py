from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (var.add(self.eps).sqrt()) + self.beta


class GSAB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_dw1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_dwd = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=6,
            dilation=3,
            groups=in_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.conv_pw = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()
        nn.init.zeros_(self.conv_pw.weight)
        if self.conv_pw.bias is not None:
            nn.init.zeros_(self.conv_pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.bn1(self.conv_dw1(x)))
        y = self.act2(self.bn2(self.conv_dwd(y)))
        y = self.conv_pw(y)
        return self.gate(y)


class LSAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn_map = torch.cat([avg_out, max_out], dim=1)
        attn_map = self.gate(self.conv(attn_map))
        return attn_map


class SAA(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.gsab = GSAB(in_channels)
        self.lsab = LSAB()

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        f_gsab = x_hat * self.gsab(x_hat)
        f_lsab = x_hat * self.lsab(x_hat)
        return f_gsab + f_lsab


class FFN(nn.Module):
    def __init__(self, in_channels: int, expansion: int = 4):
        super().__init__()
        del expansion
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.act = nn.SiLU(inplace=True)
        self.pw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.pw_conv.weight)
        if self.pw_conv.bias is not None:
            nn.init.zeros_(self.pw_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)
        x = self.act(x)
        x = self.pw_conv(x)
        return residual + x


class LSTL(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.saa = SAA(in_channels)
        self.norm2 = LayerNorm2d(in_channels)
        self.ffn = FFN(in_channels)
        self.res1 = nn.Parameter(torch.tensor(0.0))
        self.res2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.norm1(x)
        x = x + self.res1 * self.saa(x_hat)
        x_hat = self.norm2(x)
        x = x + self.res2 * self.ffn(x_hat)
        return x


class EfficientNetB0WithLSTL(nn.Module):
    def __init__(
        self,
        num_classes: int,
        use_pretrained: bool = True,
        insertion_channels: int = 112,
        probe_img_size: int = 224,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        try:
            self.base = efficientnet_b0(weights=weights)
        except Exception as exc:
            print(f"Warning: failed to load pretrained weights ({exc}). Falling back to random init.")
            self.base = efficientnet_b0(weights=None)

        in_features = self.base.classifier[1].in_features
        self.insert_after_index = None
        self.insertion_channels = insertion_channels
        with torch.no_grad():
            probe = torch.zeros(1, 3, probe_img_size, probe_img_size)
            x = probe
            for index, layer in enumerate(self.base.features):
                x = layer(x)
                channels, height, width = x.shape[1], x.shape[2], x.shape[3]
                if self.insert_after_index is None and channels == insertion_channels and height >= 14 and width >= 14:
                    self.insert_after_index = index
                    break
            if self.insert_after_index is None:
                self.insert_after_index = len(self.base.features) - 1
                self.insertion_channels = x.shape[1]

        self.lstl = LSTL(in_channels=self.insertion_channels)
        self.base.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.base.features):
            x = layer(x)
            if index == self.insert_after_index:
                x = self.lstl(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.classifier(x)
        return x
