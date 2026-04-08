from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .common import ConvBNAct, CSPStage


class PromptDetBackbone(nn.Module):
    def __init__(self, widths: List[int]):
        super().__init__()
        c1, c2, c3, c4 = widths
        self.stem = nn.Sequential(
            ConvBNAct(3, c1, 3, stride=2),
            ConvBNAct(c1, c1, 3),
        )
        self.stage1 = CSPStage(c1, c2, depth=1)
        self.stage2 = CSPStage(c2, c3, depth=2)
        self.stage3 = CSPStage(c3, c4, depth=2)
        self.stage4 = CSPStage(c4, c4, depth=2)
        self.out_channels = [c2, c3, c4, c4]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
