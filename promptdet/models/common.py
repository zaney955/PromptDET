from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 32)
        self.cv1 = ConvBNAct(channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class CSPStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super().__init__()
        self.down = ConvBNAct(in_channels, out_channels, 3, stride=2)
        blocks = [ResidualBlock(out_channels) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, out_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        out_dim = out_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
