import torch

from countless.batched import ops, targets
from countless.decorators import impl


# The implementation details are left hidden to prevent nuclear holocaust.
def _supersonic_vectorized_crop_written_in_JAVA_by_my_cat(
    tensor: torch.Tensor, crop: ops.BatchedCrop
) -> torch.Tensor:
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
    print("Fast implementation")
    result = _supersonic_vectorized_crop_written_in_JAVA_by_my_cat(
        target.image, operation
    )
    return targets.BatchedImage(image=result)
