from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .common import ConvBNAct


class PromptDetNeck(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__()
        c3, c4, c5 = in_channels
        self.lat3 = ConvBNAct(c3, out_channels, 1)
        self.lat4 = ConvBNAct(c4, out_channels, 1)
        self.lat5 = ConvBNAct(c5, out_channels, 1)

        self.fuse4 = ConvBNAct(out_channels * 2, out_channels, 3)
        self.fuse3 = ConvBNAct(out_channels * 2, out_channels, 3)
        self.down3 = ConvBNAct(out_channels, out_channels, 3, stride=2)
        self.down4 = ConvBNAct(out_channels, out_channels, 3, stride=2)
        self.out4 = ConvBNAct(out_channels * 2, out_channels, 3)
        self.out5 = ConvBNAct(out_channels * 2, out_channels, 3)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3 = self.lat3(feats["p3"])
        p4 = self.lat4(feats["p4"])
        p5 = self.lat5(feats["p5"])

        up5 = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4_td = self.fuse4(torch.cat([p4, up5], dim=1))
        up4 = F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")
        p3_td = self.fuse3(torch.cat([p3, up4], dim=1))

        p4_out = self.out4(torch.cat([p4_td, self.down3(p3_td)], dim=1))
        p5_out = self.out5(torch.cat([p5, self.down4(p4_out)], dim=1))
        return {"p3": p3_td, "p4": p4_out, "p5": p5_out}
