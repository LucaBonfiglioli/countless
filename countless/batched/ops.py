from __future__ import annotations

from dataclasses import dataclass

import torch

from countless.basic.ops import Crop
from countless.core import BatchedOperation


@dataclass
class BatchedCrop(BatchedOperation[Crop]):
    xy: torch.Tensor
    wh: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "batched_crop"

    @classmethod
    def _batch(cls, *unbatched: Crop) -> BatchedCrop:
        xy = torch.stack([op.xy for op in unbatched])
        wh = torch.stack([op.wh for op in unbatched])
        assert torch.all(wh == wh[0]), "Cannot batch crops with different sizes"
        return BatchedCrop(xy=xy, wh=wh[0])

    def unbatch(self) -> list[Crop]:
        return [Crop(xy=xy, wh=self.wh) for xy in self.xy]

    def size(self) -> int:
        return len(self.xy)
