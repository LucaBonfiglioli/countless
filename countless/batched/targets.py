from __future__ import annotations

from dataclasses import dataclass

import torch

from countless.basic.targets import Image
from countless.core import BatchedTarget


@dataclass
class BatchedImage(BatchedTarget[Image]):
    image: torch.Tensor

    @classmethod
    def name(cls) -> str:
        return "batched_image"

    @classmethod
    def _batch(cls, *unbatched: Image) -> BatchedImage:
        return BatchedImage(image=torch.stack([x.image for x in unbatched]))

    def unbatch(self) -> list[Image]:
        return [Image(image=x) for x in self.image]

    def size(self) -> int:
        return len(self.image)
