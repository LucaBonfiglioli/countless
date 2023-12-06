import torch

from countless.batched import ops, targets
from countless.decorators import impl


def _crop_spatial_batched(tensor: torch.Tensor, crop: ops.BatchedCrop) -> torch.Tensor:
    xs, ys = crop.xy.unbind(dim=-1)
    w, h = crop.wh
    tensors = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        tensors.append(tensor[i, :, y : y + h, x : x + w])
    return torch.stack(tensors)


@impl()
def crop_image(
    operation: ops.BatchedCrop, target: targets.BatchedImage
) -> targets.BatchedImage:
    result = _crop_spatial_batched(target.image, operation)
    return targets.BatchedImage(image=result)
