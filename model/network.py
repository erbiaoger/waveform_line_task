"""U-Net style binary segmentation model for waveform-line data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class UNetConfig:
    """Model configuration for the waveform-line segmenter."""

    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 4
    dropout: float = 0.0


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = max(1, min(8, out_channels // 8))
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout2d(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = int(skip.shape[-2] - x.shape[-2])
        diff_x = int(skip.shape[-1] - x.shape[-1])
        if diff_x != 0 or diff_y != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class WaveformLineUNet(nn.Module):
    """Lightweight U-Net for single-class line segmentation."""

    def __init__(self, config: Optional[UNetConfig] = None) -> None:
        super().__init__()
        self.config = config or UNetConfig()
        c = self.config
        ch = int(c.base_channels)
        self.stem = DoubleConv(int(c.in_channels), ch, dropout=float(c.dropout))
        self.down1 = DownBlock(ch, ch * 2, dropout=float(c.dropout))
        self.down2 = DownBlock(ch * 2, ch * 4, dropout=float(c.dropout))
        self.down3 = DownBlock(ch * 4, ch * 8, dropout=float(c.dropout))
        self.down4 = DownBlock(ch * 8, ch * 16, dropout=float(c.dropout))
        self.up1 = UpBlock(ch * 16, ch * 8, ch * 8, dropout=float(c.dropout))
        self.up2 = UpBlock(ch * 8, ch * 4, ch * 4, dropout=float(c.dropout))
        self.up3 = UpBlock(ch * 4, ch * 2, ch * 2, dropout=float(c.dropout))
        self.up4 = UpBlock(ch * 2, ch, ch, dropout=float(c.dropout))
        self.head = nn.Conv2d(ch, int(c.out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)

